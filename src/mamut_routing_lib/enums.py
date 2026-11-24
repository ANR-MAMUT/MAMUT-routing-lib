from enum import Enum


class BenchmarkName(str, Enum):
    SINTEF_2008 = "Sintef2008"
    DIMACS_2021 = "Dimacs2021"
    PORYOS_2026 = "Poryos2026"
    MAMUT_2026 = "Mamut2026"
    ORTEC_2022 = "Ortec2022"
    DABIA_2013 = "Dabia2013"
    ARI_2018 = "Ari2018"
    VU_2020 = "Vu2020"
    RIFKI_2020 = "Rifki2020"
    LERA_2026 = "Lera2026"
    BLAUTH_2024 = "Blauth2024"


class InstanceOrigin(str, Enum):
    SOLOMON_1987 = "Solomon1987"
    GEHHOM_1999 = "GehHom1999"
    OSM_CVRP_GEN = "OsmCvrpGen"
    ORTEC_2022 = "Ortec2022"
    ARI_2018 = "Ari2018"
    RIFKI_2020 = "Rifki2020"
    BLAUTH_2024 = "Blauth2024"


class ProblemType(str, Enum):
    CVRP = "CVRP"
    VRPTW = "VRPTW"
    TDVRPTW = "TDVRPTW"
    TDVRP = "TDVRP"


class MetricVariant(str, Enum):
    FASTEST = "fastest"
    SHORTEST = "shortest"
    EUCLIDEAN = "euclidean"


class ObjectiveFunction(str, Enum):
    HIERARCHICAL_VEHICLE_COST = "HierarchicalVehicleCost"
    MONO_COST = "MonoCost"
    DURATION = "Duration"
    FLEET_COST_DURATION = "FleetCostDuration"


class SolutionCheckStatus(str, Enum):
    """Outcome of a check: ``valid`` or the first violation met, in route order.

    The checker stops at the first violation, so a solution with several defects
    reports only one. ``route_timing_infeasible`` is reserved for the
    time-dependent checker.
    """
    VALID = "valid"
    INVALID_CUSTOMER_INDEX = "invalid_customer_index"
    CUSTOMER_SERVED_MULTIPLE_TIMES = "customer_served_multiple_times"
    VEHICLE_CAPACITY_EXCEEDED = "vehicle_capacity_exceeded"
    TIME_WINDOW_VIOLATED = "time_window_violated"
    NOT_ALL_CUSTOMERS_SERVED = "not_all_customers_served"
    TOO_MANY_VEHICLES_USED = "too_many_vehicles_used"
    OBJECTIVE_VALUE_MISMATCH = "objective_value_mismatch"
    ROUTE_TIMING_INFEASIBLE = "route_timing_infeasible"
