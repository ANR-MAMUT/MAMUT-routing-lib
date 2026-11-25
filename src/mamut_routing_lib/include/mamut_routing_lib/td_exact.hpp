// Exact TD route composition under td-binary64-exact-v1. Coordinates and
// arrivals may repeat at a FIFO jump. Each knot retains both one-sided values.
#pragma once

#include <gmpxx.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mamut_routing::td_exact {

using Q = mpq_class;

struct Knot {
    Q x, at, right;
};
using Map = std::vector<Knot>;
using WindowInput = std::optional<std::pair<double, double>>;

inline Q exact(double value) {
    if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite binary64 value");
    return Q(value);  // GMP mpq_set_d represents the finite input exactly.
}

class Arc {
    Map knots;
    std::vector<Q> slopes;

public:
    Arc(const double *xs, const double *ys, std::size_t count) {
        if (count == 0 || !xs || !ys)
            throw std::invalid_argument("invalid ATF knot arrays");
        knots.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
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
        for (std::size_t i = 1; i < knots.size(); ++i) {
            const auto &a = knots[i-1], &b = knots[i];
            slopes.emplace_back((b.at - a.right) / (b.x - a.x));
        }
    }

    const Q &start() const { return knots.front().x; }
    const Q &end() const { return knots.back().x; }

private:
    friend Map compose(const Map &, const Arc &);

    Q value(const Q &z, bool right_limit, std::size_t &cursor) const {
        while (cursor+1 < knots.size() && knots[cursor+1].x <= z)
            ++cursor;
        const auto &k = knots[cursor];
        if (z == k.x)
            return right_limit ? k.right : k.at;
        return k.right + slopes[cursor] * (z - k.x);
    }
};

// Between a and b the prefix is affine from a.right to b.at.
inline Q crossing(const Knot &a, const Knot &b, const Q &level) {
    return a.x + (level - a.right) * (b.x - a.x) / (b.at - a.right);
}

inline void restrict_upper(Map &prefix, const Q &limit) {
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
inline Map compose(const Map &prefix, const Arc &arc) {
    Map result;
    result.reserve(prefix.size());
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
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

class Customer {
    std::optional<std::pair<Q, Q>> window;
    Q service;

public:
    Customer(WindowInput bounds, double service_time) : service(exact(service_time)) {
        if (service < 0) throw std::invalid_argument("negative service");
        if (bounds) {
            auto [opens, closes] = *bounds;
            if (opens > closes) throw std::invalid_argument("inverted customer window");
            window = std::make_pair(exact(opens), exact(closes));
        }
    }

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

// The caller owns the prepared arcs. The route borrows them only for this call.
inline Map arrival_curve(const std::vector<const Arc *> &legs, const std::vector<Customer> &customers,
                         double lo, double hi, std::optional<double> close = std::nullopt) {
    if (legs.empty() || legs.size() - 1 != customers.size() ||
        std::any_of(legs.begin(), legs.end(), [](const Arc *leg) { return !leg; }))
        throw std::invalid_argument("provide one prepared arc per route leg");

    const Arc &domain = *legs.front();
    for (const Arc *leg : legs)
        if (leg->start() != domain.start() || leg->end() != domain.end())
            throw std::invalid_argument("ATFs have unequal horizons");
    const Q low = std::max(exact(lo), domain.start());
    const Q high = std::min(exact(hi), domain.end());
    const auto depot_close = close ? std::optional<Q>{exact(*close)} : std::nullopt;
    if (high < low) return {};

    Map prefix{{low, low, low}};
    if (high > low) prefix.push_back({high, high, high});
    for (std::size_t i = 0; i < legs.size(); ++i) {
        // Travel, waiting and service are nonnegative. Thus prefix values
        // on the remaining domain are already at least the common lower bound.
        restrict_upper(prefix, domain.end());
        if (prefix.empty()) return {};
        prefix = compose(prefix, *legs[i]);
        if (i < customers.size()) customers[i].visit(prefix);
    }
    if (depot_close) restrict_upper(prefix, *depot_close);
    return prefix;
}

struct Timing {
    Q duration, departure;
};

inline std::optional<Timing> route_timing(const std::vector<const Arc *> &legs,
                                           const std::vector<Customer> &customers,
                                           double lo, double hi, std::optional<double> close = std::nullopt) {
    const Map prefix = arrival_curve(legs, customers, lo, hi, close);
    if (prefix.empty()) return std::nullopt;
    Timing best{prefix.front().at - prefix.front().x, prefix.front().x};
    for (std::size_t i = 1; i < prefix.size(); ++i) {
        Q duration = prefix[i].at - prefix[i].x;
        if (duration < best.duration)
            best = {std::move(duration), prefix[i].x};
    }
    return best;
}

}  // namespace mamut_routing::td_exact
