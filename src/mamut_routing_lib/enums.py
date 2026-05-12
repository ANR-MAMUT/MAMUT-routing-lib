from enum import Enum


class BenchmarkName(str, Enum):
    SINTEF_2008 = "Sintef2008"
    DIMACS_2021 = "Dimacs2021"
    MAMUT_2026 = "Mamut2026"
    ORTEC_2022 = "Ortec2022"


class InstanceOrigin(str, Enum):
    SOLOMON_1987 = "Solomon1987"
    GEHHOM_1999 = "GehHom1999"
    OSM_CVRP_GEN = "OsmCvrpGen"
    ORTEC_2022 = "Ortec2022"


class ProblemType(str, Enum):
    CVRP = "CVRP"
    VRPTW = "VRPTW"
    TDVRP = "TDVRP"


class MetricVariant(str, Enum):
    FASTEST = "fastest"
    SHORTEST = "shortest"
    EUCLIDEAN = "euclidean"


class ObjectiveFunction(str, Enum):
    HIERARCHICAL_VEHICLE_COST = "HierarchicalVehicleCost"
    MONO_COST = "MonoCost"
