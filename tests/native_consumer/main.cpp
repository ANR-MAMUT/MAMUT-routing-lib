#include <mamut_routing_lib/td_exact.hpp>

int main() {
    namespace td = mamut_routing::td_exact;
    // Reuse both prepared arcs across the minimum, curve and infeasibility queries.
    const double x[]{0, 15}, outward[]{1, 23.5}, back[]{1, 16};
    const td::Arc outbound(x, outward, 2), inbound(x, back, 2);
    const std::vector<const td::Arc *> legs{&outbound, &inbound};
    const std::vector<td::Customer> customers{{std::pair{6., 7.}, 1.}};
    const auto timing = td::route_timing(legs, customers, 0., 15., 15.);
    if (!timing || timing->duration != td::Q(14, 3) || timing->departure != td::Q(10, 3))
        return 1;
    const auto curve = td::arrival_curve(legs, customers, 0., 15., 15.);
    if (curve.size() != 3 || curve[1].x != timing->departure || curve[1].at != 8)
        return 2;
    return td::route_timing(legs, customers, 0., 15., 7.).has_value() ? 3 : 0;
}
