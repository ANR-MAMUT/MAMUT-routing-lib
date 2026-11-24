"""Checker-owned fields used to verify an incumbent under an exact profile."""

from functools import lru_cache
from hashlib import sha256
from importlib import import_module
from pathlib import Path
import re
from platform import python_implementation, python_version

from mamut_routing_lib.exact_types import ExactEvaluation, TD_PROFILE
from mamut_routing_lib.models import BenchmarkBKS


@lru_cache(maxsize=4)
def _implementation_digest(backend: str) -> str:
    """Fingerprint the installed evaluator sources and any native kernel used."""
    modules = {
        "cpp-dyadic-prepared": ("exact_static", "_static_exact"),
        "python-fraction": ("exact_static",),
        "cpp-gmp-compose": ("exact_td", "_td_exact"),
        "python-fraction-partition": ("exact_td", "td._fraction_partition"),
    }
    digest = sha256(b"MAMUT-EVALUATOR-v1\n")
    for name in ("exact_types", *modules[backend]):
        module = import_module("mamut_routing_lib." + name)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(sha256(Path(module.__file__).read_bytes()).digest())
    return digest.hexdigest()


def routes_sha256(routes) -> str:
    """Digest of the standard's ordered route encoding."""
    digest = sha256(f"MAMUT-ROUTES-v1\n{len(routes)}\n".encode("ascii"))
    for route in routes:
        digest.update(f"{len(route)}:{','.join(map(str, route))}\n".encode("ascii"))
    return digest.hexdigest()


def standard_metadata(evaluation: ExactEvaluation) -> dict:
    """The evaluator's portion of a standard record, independent of the family manifest."""
    result = evaluation.as_dict()
    record = {
        "profile": evaluation.profile,
        "objective": evaluation.objective.value,
        "num_routes": len(evaluation.routes),
        "routes_sha256": routes_sha256(evaluation.routes),
        "authoritative_cost": result["authoritative_cost"],
        "display_overflow": result["display_overflow"],
        "evaluation": {
            "checker": "mamut-routing-lib",
            "backend": evaluation.backend,
            "implementation_sha256": _implementation_digest(evaluation.backend),
            "runtime": f"{python_implementation()} {python_version()}",
            "verification_status": "incumbent-verified",
        },
    }
    if evaluation.profile == TD_PROFILE:
        record["route_evaluations"] = result["route_evaluations"]
    return record


def evaluation_metadata(evaluation, manifest=None) -> dict:
    record = standard_metadata(evaluation)
    if manifest is None:
        return {"exact_evaluation": record}
    if manifest.objective != evaluation.objective:
        raise ValueError("Evaluation objective differs from the verified manifest")
    return {"standard": {**record, **manifest.result_identity()}}


def assert_candidate_metadata(solution, supplied, evaluation, manifest=None):
    """Check all caller-supplied numeric assertions before replacing them with checker output."""
    for key in ("standard", "exact_evaluation"):
        if key not in supplied and key not in solution.metadata:
            continue
        if key in supplied and key in solution.metadata and supplied[key] != solution.metadata[key]:
            raise ValueError(f"Conflicting supplied {key} records")
        marked = BenchmarkBKS(instance_name=solution.instance_name, objective_function=evaluation.objective,
                              routes=solution.routes, cost=solution.cost,
                              metadata={key: supplied.get(key, solution.metadata.get(key))})
        verify_stored(marked, evaluation, manifest)
        supplied.pop(key, None)


def verify_stored(bks: BenchmarkBKS, evaluation: ExactEvaluation, manifest=None) -> None:
    """Compare recomputed values with the claims of a record that names an exact profile."""
    if bks.objective_function != evaluation.objective:
        raise ValueError("Stored BKS objective does not match the instance objective.")
    for key in ("standard", "exact_evaluation"):
        if key not in bks.metadata:
            continue
        stored = bks.metadata[key]
        if not isinstance(stored, dict):
            raise ValueError(f"Invalid stored BKS metadata.{key}")
        expected = standard_metadata(evaluation)
        if key == "standard":
            if manifest is None:
                raise ValueError("A certified record requires its verified instance manifest")
            provenance = stored.get("evaluation")
            if not isinstance(provenance, dict) or any(not provenance.get(f) for f in ("checker", "backend", "runtime")):
                raise ValueError("Certified evaluation lacks a checker, backend or runtime identity")
            fingerprint = provenance.get("implementation_sha256") or provenance.get("source_sha256")
            if not isinstance(fingerprint, str) or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None:
                raise ValueError("Certified evaluation lacks an immutable checker artifact hash")
            if provenance.get("verification_status") != "incumbent-verified":
                raise ValueError("Certified evaluation lacks an incumbent verification")
            expected.update(manifest.result_identity())
            if manifest.objective != evaluation.objective:
                raise ValueError("The certified objective disagrees with the manifest")
        for field, value in expected.items():
            if field == "evaluation":
                continue  # An independent replay can use another backend.
            actual = stored.get(field)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"Stored BKS {field} does not match exact evaluation.")
        if bks.routes != [list(route) for route in evaluation.routes] or bks.cost != evaluation.display_cost:
            raise ValueError("Stored BKS routes or display cost disagree with exact evaluation.")


def preserve_claim_record(path: Path, bks: BenchmarkBKS) -> None:
    """Keep the original bytes of an annotated incumbent before displacing it."""
    if "optimality" not in bks.metadata:
        return
    original = path.read_bytes()
    archive = path.parent / ".bks-history" / (sha256(original).hexdigest() + ".json")
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        if archive.read_bytes() != original:
            raise ValueError(f"Historical BKS digest collision or damaged archive: {archive}")
    else:
        archive.write_bytes(original)
