#include <gmpxx.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

// Exact profile composition for a fixed TD route. Each knot records its
// attained arrival and right limit, which may differ at a FIFO jump.
namespace py = pybind11;
using Q = mpq_class;

struct Knot {
    Q x, at, right;
};
using Map = std::vector<Knot>;
using ArcInput = std::pair<std::vector<double>, std::vector<double>>;
using WindowInput = std::optional<std::pair<double, double>>;

Q exact(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite binary64 value");
    return Q(value);  // GMP mpq_set_d represents the finite input exactly.
}

struct Arc {
    Map knots;
    std::vector<Q> slopes;

    explicit Arc(const ArcInput &input) {
        const auto &[xs, ys] = input;
        if (xs.empty() || xs.size() != ys.size())
            throw std::invalid_argument("invalid ATF knot arrays");
        knots.reserve(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) {
            Q x = exact(xs[i]), y = exact(ys[i]);
            if (y < x || (i && (xs[i] < xs[i-1] || ys[i] < ys[i-1])))
                throw std::invalid_argument("non-FIFO or negative travel at an ATF knot");
            if (i && xs[i] == xs[i-1]) {
                knots.back().right = std::move(y);
            } else {
                knots.push_back({std::move(x), y, y});
            }
        }
        slopes.reserve(knots.size()-1);
        for (size_t i = 1; i < knots.size(); ++i) {
            const auto &a = knots[i-1], &b = knots[i];
            slopes.emplace_back((b.at - a.right) / (b.x - a.x));
        }
    }

    Q value(const Q &z, bool right_limit, size_t &cursor) const {
        while (cursor+1 < knots.size() && knots[cursor+1].x <= z)
            ++cursor;
        const auto &k = knots[cursor];
        if (z == k.x)
            return right_limit ? k.right : k.at;
        return k.right + slopes[cursor] * (z - k.x);
    }
};

// Between a and b the prefix is affine from a.right to b.at.
Q crossing(const Knot &a, const Knot &b, const Q &level) {
    return a.x + (level - a.right) * (b.x - a.x) / (b.at - a.right);
}

void restrict_upper(Map &prefix, const Q &limit) {
    const auto first_above = std::upper_bound(prefix.begin(), prefix.end(), limit,
        [](const Q &level, const Knot &k) { return level < k.at; });
    if (first_above == prefix.end()) return;
    if (first_above == prefix.begin()) {
        prefix.clear();
        return;
    }
    auto &previous = *(first_above-1);
    if (previous.right < limit) {
        Q end = crossing(previous, *first_above, limit);
        prefix.erase(first_above, prefix.end());
        prefix.push_back({std::move(end), limit, limit});
    } else {
        previous.right = previous.at;
        prefix.erase(first_above, prefix.end());
    }
}

// Both the prefix and the arc advance monotonically. Each distinct arc
// breakpoint lying strictly inside an increasing image has one preimage.
Map compose(const Map &prefix, const Arc &arc) {
    Map result;
    result.reserve(prefix.size());
    size_t cursor = 0;
    for (size_t i = 0; i < prefix.size(); ++i) {
        const auto &a = prefix[i];
        Q at = arc.value(a.at, false, cursor);
        if (i+1 == prefix.size()) {
            result.push_back({a.x, at, at});
            break;
        }
        const auto &b = prefix[i+1];
        bool increasing = a.right < b.at;
        // A prefix plateau that lands on an arc jump retains its attained
        // value throughout the plateau, including immediately after a.x.
        Q right = arc.value(a.right, increasing, cursor);
        result.push_back({a.x, std::move(at), std::move(right)});
        while (cursor+1 < arc.knots.size() && arc.knots[cursor+1].x < b.at) {
            const auto &k = arc.knots[++cursor];
            result.push_back({crossing(a, b, k.x), k.at, k.right});
        }
    }
    return result;
}

struct Customer {
    std::optional<std::pair<Q, Q>> window;
    Q service;

    void visit(Map &prefix) const {
        if (!window) {
            if (service != 0)
                for (auto &k : prefix) {
                    k.at += service;
                    k.right += service;
                }
            return;
        }
        const auto &[opens, closes] = *window;
        restrict_upper(prefix, closes);
        // Monotonicity permits at most one extra waiting breakpoint.
        auto after = std::upper_bound(prefix.begin(), prefix.end(), opens,
            [](const Q &level, const Knot &k) { return level < k.at; });
        if (after != prefix.begin() && after != prefix.end() && (after-1)->right < opens) {
            Q x = crossing(*(after-1), *after, opens);
            prefix.insert(after, {std::move(x), opens, opens});
        }
        for (auto &k : prefix) {
            k.at = std::max(k.at, opens) + service;
            k.right = std::max(k.right, opens) + service;
        }
    }
};

py::int_ python_integer(const mpz_class &z) {
    std::string digits = z.get_str(16);
    auto value = PyLong_FromString(digits.data(), nullptr, 16);
    if (value == nullptr) throw py::error_already_set();
    return py::reinterpret_steal<py::int_>(value);
}

py::tuple fraction_pair(const Q &v) {
    return py::make_tuple(python_integer(v.get_num()), python_integer(v.get_den()));
}

class PreparedRoute {
    std::vector<Arc> arcs;
    std::vector<Customer> customers;
    Q low, high;
    std::optional<Q> depot_close;

public:
    PreparedRoute(const std::vector<ArcInput> &inputs, const std::vector<WindowInput> &windows,
                  const std::vector<double> &services, double lo, double hi,
                  std::optional<double> close)
        : low(exact(lo)), high(exact(hi)) {
        if (inputs.size() != services.size()+1 || windows.size() != services.size())
            throw std::invalid_argument("provide one arc per leg and one window/service per customer");
        arcs.reserve(inputs.size());
        for (const auto &input : inputs) {
            const Arc &arc = arcs.emplace_back(input);
            if (arc.knots.front().x != arcs.front().knots.front().x ||
                arc.knots.back().x != arcs.front().knots.back().x)
                throw std::invalid_argument("ATFs have unequal horizons");
        }
        low = std::max(low, arcs.front().knots.front().x);
        high = std::min(high, arcs.front().knots.back().x);
        if (close) depot_close = exact(*close);
        for (size_t i = 0; i < services.size(); ++i) {
            Customer customer{std::nullopt, exact(services[i])};
            if (customer.service < 0) throw std::invalid_argument("negative service");
            if (windows[i]) {
                auto [opens, closes] = *windows[i];
                if (opens > closes) throw std::invalid_argument("inverted customer window");
                customer.window = std::make_pair(exact(opens), exact(closes));
            }
            customers.push_back(std::move(customer));
        }
    }

    py::object solve() const {
        if (high < low) return py::none();
        Map prefix{{low, low, low}};
        if (high > low) prefix.push_back({high, high, high});
        for (size_t i = 0; i < arcs.size(); ++i) {
            // Travel, waiting and service are nonnegative, so all prefix
            // values on the remaining domain are already >= the common h0.
            restrict_upper(prefix, arcs[i].knots.back().x);
            if (prefix.empty()) return py::none();
            prefix = compose(prefix, arcs[i]);
            if (i < customers.size()) customers[i].visit(prefix);
        }
        if (depot_close) restrict_upper(prefix, *depot_close);
        if (prefix.empty()) return py::none();
        Q best = prefix.front().at - prefix.front().x;
        size_t earliest = 0;
        for (size_t i = 1; i < prefix.size(); ++i) {
            Q duration = prefix[i].at - prefix[i].x;
            if (duration < best) {
                best = std::move(duration);
                earliest = i;
            }
        }
        return py::make_tuple(fraction_pair(best), fraction_pair(prefix[earliest].x));
    }


};

PYBIND11_MODULE(_td_exact, m) {
    py::class_<PreparedRoute>(m, "PreparedRoute")
        .def(py::init<const std::vector<ArcInput>&, const std::vector<WindowInput>&,
                      const std::vector<double>&, double, double, std::optional<double>>(),
             py::arg("arcs"), py::arg("windows"), py::arg("services"), py::arg("lo"),
             py::arg("hi"), py::arg("depot_close") = py::none())
        .def("solve", &PreparedRoute::solve);
}
