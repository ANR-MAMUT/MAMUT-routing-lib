from enum import Enum


class BenchmarkName(str, Enum):
    SINTEF_2008 = "Sintef2008"
    DIMACS_2021 = "Dimacs2021"
    PORYOS_2026 = "Poryos2026"
    ORTEC_2022 = "Ortec2022"
    DABIA_2013 = "Dabia2013"
    ARI_2018 = "Ari2018"
    VU_2020 = "Vu2020"
    RIFKI_2020 = "Rifki2020"
    LERA_2026 = "Lera2026"


class InstanceOrigin(str, Enum):
    SOLOMON_1987 = "Solomon1987"
    GEHHOM_1999 = "GehHom1999"
    OSM_CVRP_GEN = "OsmCvrpGen"
    ORTEC_2022 = "Ortec2022"
    ARI_2018 = "Ari2018"
    RIFKI_2020 = "Rifki2020"


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
