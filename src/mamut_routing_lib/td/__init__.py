"""Time-dependent routing (TDVRPTW / TDVRP) contract and canonical checker.

See the TD benchmark standard in the MAMUT-routing repository for the format
specification: instance ``.vrp.json`` files reference an ATF sidecar
(``.atf.json`` / ``.atf.json.gz``) holding one arrival-time NDCPWLF per arc.
"""

from mamut_routing_lib.td.artifacts import (
    ATF_GZIP_SUFFIX,
    ATF_PLAIN_SUFFIX,
    ATFFormatError,
    InstanceATFs,
    LoadedTDInstance,
    atfs_to_canonical_json_bytes,
    compute_atf_sha256,
    get_atf_path_for_instance,
    load_instance_atfs,
    load_td_benchmark_instance,
    load_td_instance,
    save_instance_atfs,
    td_instance_from_payload,
)
from mamut_routing_lib.td.checker import (
    TDRouteEvaluation,
    TDSolutionCheckResult,
    canonical_route_order,
    check_td_solution,
    compute_route_duration,
    compute_route_ready_time_function,
    compute_solution_cost,
)
from mamut_routing_lib.td.models import (
    ATF_FORMAT,
    ATF_FORMAT_VERSION,
    TD_ATF_MODEL,
    AnyTDBenchmarkInstance,
    BenchmarkInstanceTDVRP,
    BenchmarkInstanceTDVRPTW,
    TDArrivalFunctionsRef,
)
from mamut_routing_lib.td.pwlf import NDCPWLF, PWLFError, make_theta

__all__ = [
    "ATF_FORMAT",
    "ATF_FORMAT_VERSION",
    "ATF_GZIP_SUFFIX",
    "ATF_PLAIN_SUFFIX",
    "ATFFormatError",
    "AnyTDBenchmarkInstance",
    "BenchmarkInstanceTDVRP",
    "BenchmarkInstanceTDVRPTW",
    "InstanceATFs",
    "LoadedTDInstance",
    "NDCPWLF",
    "PWLFError",
    "TDArrivalFunctionsRef",
    "TDRouteEvaluation",
    "TDSolutionCheckResult",
    "TD_ATF_MODEL",
    "atfs_to_canonical_json_bytes",
    "canonical_route_order",
    "check_td_solution",
    "compute_atf_sha256",
    "compute_route_duration",
    "compute_route_ready_time_function",
    "compute_solution_cost",
    "get_atf_path_for_instance",
    "load_instance_atfs",
    "load_td_benchmark_instance",
    "load_td_instance",
    "make_theta",
    "save_instance_atfs",
    "td_instance_from_payload",
]
