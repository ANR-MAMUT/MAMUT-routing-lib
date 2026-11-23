// Prepared binary64-exact-v1 arithmetic. Route coverage and capacity are
// validated by the public evaluator before this private kernel is called.
#include <boost/multiprecision/cpp_int.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <ios>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using Small = boost::multiprecision::uint128_t;
using Big = boost::multiprecision::cpp_int;
const Small small_max = (Small(1) << 127) - 1;

struct Dyadic {
    std::uint64_t mantissa;
    int exponent;
};

Dyadic decompose(double value) {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53);
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto field = (bits >> 52) & 0x7ff;
    auto mantissa = bits & ((std::uint64_t(1) << 52) - 1);
    if (field == 0 && mantissa == 0) return {0, 0};  // Includes negative zero.
    if (field == 0x7ff || (bits >> 63)) {
        throw std::invalid_argument("expected a finite nonnegative binary64 operand");
    }
    if (field) mantissa |= std::uint64_t(1) << 52;
    const int exponent = field ? static_cast<int>(field) - 1075 : -1074;
    const int zeros = std::countr_zero(mantissa);
    return {mantissa >> zeros, exponent + zeros};
}

template<class T>
T units(double value, int scale) {
    const auto [mantissa, exponent] = decompose(value);
    if (mantissa == 0) return T(0);
    const int shift = exponent + scale;
    if (shift < 0) throw std::logic_error("operand is outside the prepared dyadic scale");
    if constexpr (std::is_same_v<T, Small>) {
        if (shift + std::bit_width(mantissa) > 127)
            throw std::overflow_error("scaled value exceeds the checked integer width");
    }
    return T(mantissa) << shift;
}

Big add(const Big &a, const Big &b) { return a + b; }
Small add(const Small &a, const Small &b) {
    if (b > small_max - a) throw std::overflow_error("checked static accumulator overflow");
    return a + b;
}

// The validated Python temporal vectors contain binary64 values. Matrix
// elements arrive from the original instance and still need a numeric check.
double arc_value(py::handle raw) {
    if (!(PyLong_CheckExact(raw.ptr()) || PyFloat_CheckExact(raw.ptr())))
        throw std::invalid_argument("arc costs must be numbers");
    const double value = PyFloat_AsDouble(raw.ptr());
    if (PyErr_Occurred()) {
        if (!PyErr_ExceptionMatches(PyExc_OverflowError)) throw py::error_already_set();
        PyErr_Clear();
        throw std::invalid_argument("arc cost exceeds the binary64 input range");
    }
    return value;
}

template<class T>
struct Tables {
    std::vector<T> matrix, opens, closes, services;
};

class StaticModel {
    std::size_t vertices;
    int scale = 0;
    std::variant<Tables<Small>, Tables<Big>> tables;

    template<class T>
    void prepare(const std::vector<double> &matrix, const std::vector<double> &opens,
                 const std::vector<double> &closes, const std::vector<double> &services) {
        auto &target = tables.emplace<Tables<T>>();
        auto convert = [this](const auto &raw, auto &out) {
            out.reserve(raw.size());
            for (double value : raw) out.push_back(units<T>(value, scale));
        };
        convert(matrix, target.matrix);
        convert(opens, target.opens);
        convert(closes, target.closes);
        convert(services, target.services);
    }

    template<class Acc, class T>
    std::optional<Big> sum(const Tables<T> &data,
                           const std::vector<std::vector<std::size_t>> &routes) const {
        Acc total = 0;
        const bool temporal = !data.opens.empty();
        for (const auto &route : routes) {
            std::size_t previous = 0;
            Acc departure = temporal ? Acc(data.opens[0]) : Acc(0);
            for (std::size_t i = 0; i <= route.size(); ++i) {
                const std::size_t v = i == route.size() ? 0 : route[i];
                if (v >= vertices) throw std::invalid_argument("customer index out of range");
                const Acc arc(data.matrix[previous * vertices + v]);
                total = add(total, arc);
                if (temporal) {
                    const Acc start = std::max(add(departure, arc), Acc(data.opens[v]));
                    if (start > data.closes[v]) return std::nullopt;
                    if (v) departure = add(start, Acc(data.services[v]));
                }
                previous = v;
            }
        }
        return Big(total);
    }

public:
    StaticModel(std::size_t num_vertices, const py::sequence &rows,
                const std::vector<double> &opens, const std::vector<double> &closes,
                const std::vector<double> &services) : vertices(num_vertices) {
        if (!vertices || rows.size() != vertices ||
            vertices > std::numeric_limits<std::size_t>::max() / vertices)
            throw std::invalid_argument("invalid arc matrix dimensions");
        if ((!opens.empty() && opens.size() != vertices) ||
            closes.size() != opens.size() || services.size() != opens.size())
            throw std::invalid_argument("invalid temporal vector dimensions");

        double max_arc = 0;
        std::vector<double> matrix;
        matrix.reserve(vertices * vertices);
        auto inspect = [this](double value) {
            const auto parts = decompose(value);
            scale = std::max(scale, -parts.exponent);
        };
        for (std::size_t i = 0; i < vertices; ++i) {
            const auto row_object = rows[i];
            if (!(PyList_Check(row_object.ptr()) || PyTuple_Check(row_object.ptr())))
                throw std::invalid_argument("expected an arc matrix row");
            const auto row = py::reinterpret_borrow<py::sequence>(row_object);
            if (row.size() != vertices) throw std::invalid_argument("invalid arc matrix dimensions");
            for (std::size_t j = 0; j < vertices; ++j) {
                const double value = arc_value(row[j]);
                inspect(value);
                if (i == j && value != 0) throw std::invalid_argument("nonzero arc matrix diagonal");
                max_arc = std::max(max_arc, value);
                matrix.push_back(value);
            }
        }
        for (const auto *field : {&opens, &closes, &services}) {
            for (double x : *field) inspect(x);
        }
        const auto maximum = [](const auto &field) {
            return field.empty() ? 0.0 : *std::max_element(field.begin(), field.end());
        };
        const Big arc = units<Big>(max_arc, scale);
        const Big objective_bound = 2 * Big(vertices - 1) * arc;
        const Big time_bound = units<Big>(maximum(opens), scale) +
                               vertices * (arc + units<Big>(maximum(services), scale));
        if (std::max(objective_bound, time_bound) <= Big(small_max) &&
            units<Big>(maximum(closes), scale) <= Big(small_max)) {
            prepare<Small>(matrix, opens, closes, services);
        } else {
            prepare<Big>(matrix, opens, closes, services);
        }
    }

    py::object evaluate(const std::vector<std::vector<std::size_t>> &routes) const {
        std::optional<Big> total;
        {
            py::gil_scoped_release release;
            if (const auto *narrow = std::get_if<Tables<Small>>(&tables)) {
                try {
                    total = sum<Small>(*narrow, routes);
                } catch (const std::overflow_error &) {
                    total = sum<Big>(*narrow, routes);
                }
            } else {
                total = sum<Big>(std::get<Tables<Big>>(tables), routes);
            }
        }
        if (!total) return py::none();
        // Hex conversion has no decimal string digit limit in CPython.
        std::string hex = total->str(0, std::ios_base::hex);
        auto integer = py::reinterpret_steal<py::int_>(PyLong_FromString(hex.data(), nullptr, 16));
        if (!integer) throw py::error_already_set();
        return py::make_tuple(std::move(integer), scale);
    }
};

PYBIND11_MODULE(_static_exact, module) {
    py::class_<StaticModel>(module, "StaticModel")
        .def(py::init<std::size_t, const py::sequence &, const std::vector<double> &,
                      const std::vector<double> &, const std::vector<double> &>())
        .def("evaluate", &StaticModel::evaluate);
}
