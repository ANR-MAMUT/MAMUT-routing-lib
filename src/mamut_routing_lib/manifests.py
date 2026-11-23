"""Verify the files and complete binary64 operands belonging to a published edition."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import bz2
import gzip
import lzma
from hashlib import file_digest, sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import re
from struct import pack

from mamut_routing_lib.artifacts import hydrate_collection_instance, instance_problem_type, load_benchmark_instance
from mamut_routing_lib.enums import ObjectiveFunction
from mamut_routing_lib.exact_types import STATIC_PROFILE, TD_PROFILE
from mamut_routing_lib.json_utils import strict_json_loads
from mamut_routing_lib.models import ArcCostsDistancesRef, ArcCostsEuclidean, LoadedStaticInstance
from mamut_routing_lib.objectives import base_objective
from mamut_routing_lib.sidecars import find_collection_root, load_collection_marker, validate_relative_sidecar_path
from mamut_routing_lib.td.artifacts import LoadedTDInstance
from mamut_routing_lib.td.manifest_stream import stream_td_materialization
from mamut_routing_lib.td.reprice import route_arcs
from mamut_routing_lib.td.models import AnyTDBenchmarkInstance, TDIGPProfileRef, TDRoadGraphRef


SPECIFICATION = "mamut-routing/0.1-review"


def manifest_path_for(instance_path: str | Path) -> Path:
    path = Path(instance_path)
    if path.name.endswith(".vrp.json"):
        return path.with_name(path.name.removesuffix(".vrp.json") + ".manifest.json")
    return path.with_suffix(".manifest.json")


def manifest_bytes(manifest: Mapping) -> bytes:
    """Serialize a new edition deterministically. Readers hash the actual published bytes."""
    return (json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def file_sha256(path: str | Path, scope: str = "file") -> str:
    if scope not in ("file", "decompressed"):
        raise ValueError(f"Unknown hash scope {scope!r}")
    path = Path(path)
    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(path.suffix, open) if scope == "decompressed" else open
    with opener(path, "rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _operand(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Expected a finite numerical operand")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("Operand exceeds the binary64 range") from exc
    if not isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative binary64 operand")
    return value or 0.0


def matrix_sha256(matrix) -> str:
    """Hash the complete row-major MAMUT-MATRIX-v1 stream, including the diagonal."""
    n = len(matrix)
    if not n:
        raise ValueError("The matrix must include a depot")
    digest = sha256(f"MAMUT-MATRIX-v1\nbinary64\n{n}\n".encode("ascii"))
    for i, row in enumerate(matrix):
        if len(row) != n or _operand(row[i]) != 0:
            raise ValueError("Expected a square matrix with a zero diagonal")
        digest.update(pack(f">{n}d", *map(_operand, row)))
    return digest.hexdigest()


def atf_stream_sha256(num_vertices: int, horizon, arcs: Iterable) -> str:
    """Hash complete (tail, head, xs, ys) records supplied in lexicographic arc order."""
    if type(num_vertices) is not int or num_vertices < 1:
        raise ValueError("Expected a positive vertex count including the depot")
    low, high = map(_operand, horizon)
    if low >= high:
        raise ValueError("Expected a nonempty TD horizon")
    digest = sha256(f"MAMUT-ATF-v1\nbinary64\n{num_vertices}\n".encode("ascii"))
    digest.update(pack(">dd", low, high))
    records = iter(arcs)
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i == j:
                continue
            arc = next(records, None)
            if arc is None:
                raise ValueError(f"Missing directed arc {(i, j)}")
            tail, head, xs, ys = arc
            if type(tail) is not int or type(head) is not int or (tail, head) != (i, j):
                raise ValueError(f"Expected directed arc {(i, j)}")
            if len(xs) < 2 or len(xs) != len(ys) or (_operand(xs[0]), _operand(xs[-1])) != (low, high):
                raise ValueError(f"Invalid knot sequence on arc {(i, j)}")
            digest.update(f"{i},{j},{len(xs)}\n".encode("ascii"))
            previous_x = previous_y = low
            for x, y in zip(xs, ys):
                x, y = _operand(x), _operand(y)
                if x < previous_x or y < previous_y or y < x:
                    raise ValueError(f"Invalid FIFO arrival function on arc {(i, j)}")
                digest.update(pack(">dd", x, y))
                previous_x, previous_y = x, y
    if next(records, None) is not None:
        raise ValueError("Extra arc beyond the complete directed graph")
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifiedManifest:
    specification: str
    benchmark_revision: str
    manifest_sha256: str
    objective: ObjectiveFunction
    checkable: LoadedStaticInstance | LoadedTDInstance

    def result_identity(self) -> dict[str, str]:
        return dict(specification=self.specification, benchmark_revision=self.benchmark_revision,
                    manifest_sha256=self.manifest_sha256)


def _required_text(data, key):
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid manifest field {key!r}")
    return value


def _artifact(root, reference):
    if not isinstance(reference, dict):
        raise ValueError("Expected an artifact reference in the manifest")
    name = validate_relative_sidecar_path(_required_text(reference, "path"))
    if Path(name).as_posix() != name or name == ".":
        raise ValueError(f"Expected a canonical artifact path, got {name!r}")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact resolves outside the family root: {name}")
    expected = _required_text(reference, "sha256")
    if re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise ValueError(f"Invalid SHA-256 for {name}")
    if (".git" in path.relative_to(root).parts or ".bks." in path.name or path.name.endswith(".manifest.json")):
        raise ValueError(f"Certification input must be independent of the result and edition manifest: {name}")
    if path.suffix in (".gz", ".bz2", ".xz") and "scope" not in reference:
        raise ValueError(f"Compressed artifact {name} must declare its hash scope")
    if file_sha256(path, reference.get("scope", "file")) != expected:
        raise ValueError(f"Manifest SHA-256 mismatch for {name}")
    return path


def _blauth_parameters(instance, declaration):
    upstream = instance.metadata["generator"].get("upstream_files", {})
    members = declaration.get("upstream_files")
    if not upstream or not isinstance(members, list) or len(members) != len(upstream):
        raise ValueError("The Blauth recipe must identify every original source member")
    seen = {}
    for member in members:
        if not isinstance(member, dict):
            raise ValueError("Invalid Blauth source member")
        path = validate_relative_sidecar_path(_required_text(member, "path"))
        archive_member = validate_relative_sidecar_path(_required_text(member, "archive_member"))
        if not archive_member.endswith("/" + path) or member.get("scope") != "file":
            raise ValueError("Invalid Blauth archive member or byte scope")
        _required_text(member, "uri")
        seen[path] = _required_text(member, "sha256")
    if seen != {"instances/" + name: digest for name, digest in upstream.items()}:
        raise ValueError("Blauth source member digests disagree with the descriptor")
    return {"city": instance.instance_name.removeprefix("Blauth-"), "n": instance.num_customers,
            "upstream_root": ".mamut/external/vrptdt-benchmark", "upstream_files": members,
            "upstream_commit_from": "metadata.generator.upstream_commit", "output_root": ".",
            "output_path": instance.td.atf_path, "output_decompressed_sha256": instance.td.atf_sha256}



def _materialization(instance, root, sidecar_names, material):
    td = isinstance(instance, AnyTDBenchmarkInstance)
    kind = _required_text(material, "kind")
    required = []
    parameters = {}
    callable_name = None
    if td:
        recipe = instance.td
        if isinstance(recipe, TDIGPProfileRef):
            kinds, required = {"regenerated"}, [recipe.categories_path]
            callable_name = "mamut_routing_lib.td.igp.materialize_instance_atfs"
            parameters = {"time_periods": [list(period) for period in recipe.time_periods], "speeds": recipe.speeds,
                          "categories_path": recipe.categories_path}
        elif isinstance(recipe, TDRoadGraphRef):
            kinds, required = {"regenerated"}, [recipe.graph.path, recipe.traffic.path]
            callable_name = "mamut_routing_lib.td.roadgraph.materialize_instance_atfs_roadgraph"
            parameters = {"sample_step": recipe.sample_step, "simplify_tolerance": recipe.simplify_tolerance,
                          "graph_path": recipe.graph.path, "traffic_path": recipe.traffic.path}
        else:
            kinds = {"stored-atf", "regenerated"}
            if kind == "stored-atf":
                required = [recipe.atf_path]
    else:
        source = getattr(instance, "arc_costs_source", None)
        if isinstance(source, ArcCostsEuclidean):
            kinds = {"regenerated"}
            callable_name = "mamut_routing_lib.artifacts.resolve_arc_costs"
            parameters = {"model": "euclidean", "decimals": source.decimals}
        elif isinstance(source, ArcCostsDistancesRef):
            kinds = {"sidecar-matrix", "regenerated"}
            if kind == "sidecar-matrix":
                required = [source.distances.path]
            elif instance.benchmark_name.value == "Mamut2026":
                road = Path(source.distances.path).with_name(instance.instance_name + ".road.json.gz")
                required = [road.as_posix()]
                callable_name = "mamut_routing_tools.family.materialize_distances"
                parameters = {"metric": instance.metric_variant.value, "decimals": 3,
                              "output_path": source.distances.path, "output_decompressed_sha256": source.distances.sha256}
        else:
            kinds = {"embedded-matrix"}
    if kind not in kinds or material.get("interpretation") != "binary64":
        raise ValueError("Manifest materialization disagrees with the descriptor or profile")
    if not set(required).issubset(sidecar_names):
        raise ValueError("Manifest omits a sidecar required by the descriptor")
    if kind == "regenerated":
        recipe = material.get("recipe")
        if not isinstance(recipe, dict) or not isinstance(recipe.get("parameters"), dict):
            raise ValueError("A regenerated materialization requires a recipe and parameters")
        actual_callable = _required_text(recipe, "callable")
        if td and instance.td.model == "atf-ndcpwlf" and instance.benchmark_name.value == "Blauth2024":
            callable_name = "mamut_routing_tools.conversion.blauth2024.convert_instance"
            parameters = _blauth_parameters(instance, recipe["parameters"])
        if callable_name is not None and (actual_callable != callable_name or recipe["parameters"] != parameters):
            raise ValueError("Manifest recipe differs from the published descriptor or materializer")
        archives = recipe.get("artifacts")
        if not isinstance(archives, list) or not archives:
            raise ValueError("A regenerated materialization requires archived implementation and environment artifacts")
        for reference in archives:
            _artifact(root, reference)
    return "atf_stream_sha256" if td else "matrix_sha256"


def verify_manifest(instance_path: str | Path, *, expected_sha256: str | None = None,
                    routes: Iterable[Iterable[int]] | None = None, atf_cache_root: Path | None = None) -> VerifiedManifest:
    """Verify the adjacent manifest, required artifact files and complete materialization without Git.

    Materialize omitted sidecars with the published family recipe before loading them.
    For TD, pass routes to retain only their arcs, or an empty iterable for digest-only
    verification. The default keeps all arcs for subsequent route evaluations. Every
    mode verifies the entire graph. A cache may supply a regenerated ATF sidecar.
    """
    instance_path = Path(instance_path).resolve()
    manifest_path = manifest_path_for(instance_path)
    raw = manifest_path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Manifest SHA-256 mismatch for {manifest_path}")
    manifest = strict_json_loads(raw)
    if not isinstance(manifest, dict) or manifest.get("specification") != SPECIFICATION:
        raise ValueError("Unsupported manifest specification")
    revision = _required_text(manifest, "benchmark_revision")
    collection = find_collection_root(instance_path)
    root = collection or instance_path.parent
    if _artifact(root, manifest.get("instance")) != instance_path:
        raise ValueError("Manifest identifies a different descriptor")
    instance = load_benchmark_instance(instance_path)
    family = instance.benchmark_name.value
    if collection and load_collection_marker(root / "mamut-collection.json").family != family:
        raise ValueError("Family marker disagrees with the instance")
    td = isinstance(instance, AnyTDBenchmarkInstance)
    expected = dict(family=family, instance_name=instance.instance_name, problem=instance_problem_type(instance).value,
                    default_profile=TD_PROFILE if td else STATIC_PROFILE, objective=base_objective(instance).value)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {key} disagrees with the descriptor or published objective")
    units = manifest.get("units")
    if not isinstance(units, dict) or not isinstance(units.get("cost"), str) or not units["cost"].strip():
        raise ValueError("Manifest must declare the cost units")
    features = manifest.get("features")
    if not isinstance(features, list) or not all(isinstance(f, str) and f for f in features):
        raise ValueError("Manifest must declare the supported features")
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, list):
        raise ValueError("Manifest must list the required sidecars")
    names = set()
    for ref in sidecars:
        path = _artifact(root, ref)
        if path in (instance_path, manifest_path) or ref["path"] in names:
            raise ValueError("Repeated or circular manifest artifact")
        names.add(ref["path"])
    material = manifest.get("materialization")
    if not isinstance(material, dict):
        raise ValueError("Manifest must declare a materialization")
    field = _materialization(instance, root, names, material)
    if td:
        selected = None
        if routes is not None:
            selected = {(i, j) for i, j in route_arcs(routes) if i != j
                        and type(i) is int and type(j) is int and 0 <= i <= instance.num_customers
                        and 0 <= j <= instance.num_customers}
        streamed = stream_td_materialization(instance_path, root, selected, atf_cache_root=atf_cache_root)
        if "canonical_atf_sha256" in material and material["canonical_atf_sha256"] != streamed.canonical_sha256:
            raise ValueError("Manifest canonical_atf_sha256 disagrees with the ATF materialization")
        td_source = instance.td
        atf_path = root / td_source.atf_path if hasattr(td_source, "atf_path") else None
        if atf_path is not None and not atf_path.exists() and atf_cache_root is not None:
            atf_path = atf_cache_root / td_source.atf_path
        checkable = LoadedTDInstance(instance, streamed.operands, instance_path, atf_path,
                                     categories_path=root / td_source.categories_path if isinstance(td_source, TDIGPProfileRef) else None,
                                     road_graph_path=root / td_source.graph.path if isinstance(td_source, TDRoadGraphRef) else None,
                                     traffic_path=root / td_source.traffic.path if isinstance(td_source, TDRoadGraphRef) else None,
                                     collection_root=collection)
        actual = streamed.atf_stream_sha256
    else:
        checkable = hydrate_collection_instance(instance, instance_path)
        actual = matrix_sha256(checkable.arc_costs)
    if material.get(field) != actual:
        raise ValueError(f"Manifest {field} disagrees with the complete materialization")
    return VerifiedManifest(SPECIFICATION, revision, digest, base_objective(instance), checkable)


def adjacent_manifest(instance_path, objective=None, *, routes=None) -> VerifiedManifest | None:
    if not manifest_path_for(instance_path).exists():
        return None
    verified = verify_manifest(instance_path, routes=routes)
    if objective is not None and objective != verified.objective:
        return None  # The alternate objective is an unbound evaluation.
    return verified
