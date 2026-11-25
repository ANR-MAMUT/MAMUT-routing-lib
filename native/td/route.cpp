#include <mamut_routing_lib/td_exact.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py = pybind11;
namespace td = mamut_routing::td_exact;

namespace {
using td::Q;
using ArcInput = std::pair<std::vector<double>, std::vector<double>>;

py::int_ python_integer(const mpz_class &z) {
    std::string digits = z.get_str(16);
    auto value = PyLong_FromString(digits.data(), nullptr, 16);
    if (value == nullptr) throw py::error_already_set();
    return py::reinterpret_steal<py::int_>(value);
}

py::tuple fraction_pair(const Q &v) {
    return py::make_tuple(python_integer(v.get_num()), python_integer(v.get_den()));
}

std::shared_ptr<td::Arc> prepare_arc(const ArcInput &input) {
    const auto &[xs, ys] = input;
    if (xs.size() != ys.size()) throw std::invalid_argument("invalid ATF knot arrays");
    return std::make_shared<td::Arc>(xs.data(), ys.data(), xs.size());
}

class PreparedRoute {
    std::vector<std::shared_ptr<td::Arc>> storage;
    std::vector<const td::Arc *> legs;
    std::vector<td::Customer> customers;
    double lo, hi;
    std::optional<double> close;

public:
    PreparedRoute(std::vector<std::shared_ptr<td::Arc>> arcs, const std::vector<td::WindowInput> &windows,
                  const std::vector<double> &services, double lo, double hi, std::optional<double> close)
        : storage(std::move(arcs)), lo(lo), hi(hi), close(close) {
        if (storage.size() != services.size()+1 || windows.size() != services.size())
            throw std::invalid_argument("provide one arc per leg and one window/service per customer");
        for (const auto &arc : storage) legs.push_back(arc.get());
        for (std::size_t i = 0; i < services.size(); ++i) customers.emplace_back(windows[i], services[i]);
    }

    py::object solve() const {
        const auto timing = td::route_timing(legs, customers, lo, hi, close);
        if (!timing) return py::none();
        return py::make_tuple(fraction_pair(timing->duration), fraction_pair(timing->departure));
    }

    py::object curve() const {
        const auto prefix = td::arrival_curve(legs, customers, lo, hi, close);
        if (prefix.empty()) return py::none();
        py::tuple result(prefix.size());
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            const auto &k = prefix[i];
            result[i] = py::make_tuple(fraction_pair(k.x), fraction_pair(k.at), fraction_pair(k.right));
        }
        return result;
    }
};
}  // namespace

PYBIND11_MODULE(_td_exact, m) {
    py::class_<td::Arc, std::shared_ptr<td::Arc>>(m, "PreparedArc").def(py::init(&prepare_arc));
    py::class_<PreparedRoute>(m, "PreparedRoute")
        .def(py::init<std::vector<std::shared_ptr<td::Arc>>, const std::vector<td::WindowInput>&,
                      const std::vector<double>&, double, double, std::optional<double>>(),
             py::arg("arcs"), py::arg("windows"), py::arg("services"), py::arg("lo"),
             py::arg("hi"), py::arg("depot_close") = py::none())
        .def(py::init([](const std::vector<ArcInput> &inputs, const std::vector<td::WindowInput> &windows,
                         const std::vector<double> &services, double lo, double hi, std::optional<double> close) {
                 std::vector<std::shared_ptr<td::Arc>> arcs;
                 for (const auto &input : inputs) arcs.push_back(prepare_arc(input));
                 return PreparedRoute(std::move(arcs), windows, services, lo, hi, close);
             }), py::arg("arcs"), py::arg("windows"), py::arg("services"), py::arg("lo"),
             py::arg("hi"), py::arg("depot_close") = py::none())
        .def("solve", &PreparedRoute::solve)
        .def("curve", &PreparedRoute::curve);
}
