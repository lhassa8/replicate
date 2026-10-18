"""Read back a CPU export without executing returned code or model loaders."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from .export_math import check_merge, compare_native_probabilities
from .export_worker import LIMITS, MODEL_RESULTS
from .gguf_export import identity
from .safe_archive import BoundedReader, safe_relative
from .training_inputs import load, require


def read_archive(path, expected, *, destination=None):
    """Inspect first; optional extraction writes only to a new private directory."""
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "Expected a regular result archive")
    require(0 < expected["bytes"] <= LIMITS["maximum_result_bytes"]
            and identity(path) == {key: expected[key] for key in ("bytes", "sha256")},
            "Returned archive identity differs")
    if destination is not None:
        destination = Path(destination)
        require(not destination.exists() and not destination.is_symlink(), "Extraction target must be new")
        destination.mkdir(mode=0o700)
    files, names, folded, components, payload, metadata = {}, set(), set(), {}, 0, 0
    retention = None
    with path.open("rb") as raw, gzip.GzipFile(fileobj=raw) as decoded:
        bounded = BoundedReader(decoded, LIMITS["maximum_result_bytes"] + 64 * 1024**2)
        with tarfile.open(fileobj=bounded, mode="r|", encoding="utf-8", errors="strict") as archive:
            for member in archive:
                name = safe_relative(member.name)
                require(member.isfile() and member.sparse is None and not member.mode & 0o7000,
                        "Result archive contains an unsupported entry")
                require(name not in names and name.casefold() not in folded
                        and len(names) < LIMITS["maximum_result_files"], "Duplicate or excessive result entries")
                require(not any(name.casefold().startswith(prior + "/") or prior.startswith(name.casefold() + "/")
                                for prior in folded), "Result file paths overlap")
                parts = name.split("/")
                for index in range(1, len(parts) + 1):
                    component = "/".join(parts[:index])
                    require(components.setdefault(component.casefold(), component) == component,
                            "Result paths contain case aliases")
                names.add(name)
                folded.add(name.casefold())
                maximum = (LIMITS["maximum_q4_file_bytes"] if name in MODEL_RESULTS
                           else LIMITS["maximum_metadata_file_bytes"])
                require(0 <= member.size <= maximum, "Returned file exceeds its bound")
                payload += member.size
                if name not in MODEL_RESULTS:
                    metadata += member.size
                require(payload <= LIMITS["maximum_result_bytes"]
                        and metadata <= LIMITS["maximum_metadata_bytes"], "Returned payload exceeds its bound")
                digest, remaining, captured = hashlib.sha256(), member.size, bytearray()
                target = None
                if destination is not None:
                    output = destination / name
                    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    target = output.open("xb")
                try:
                    with archive.extractfile(member) as content:
                        while remaining:
                            block = content.read(min(remaining, 1024**2))
                            require(bool(block), "Truncated result file")
                            remaining -= len(block)
                            digest.update(block)
                            if name == "retention.json":
                                captured.extend(block)
                            if target is not None:
                                target.write(block)
                    files[name] = {"bytes": member.size, "sha256": digest.hexdigest()}
                finally:
                    if target is not None:
                        target.close()
                        output.chmod(0o600)
                if name == "retention.json":
                    retention = json.loads(captured)
            while bounded.read(65536):
                pass
    require(retention is not None and retention.get("schema") == "organism-cpu-export-retention-v1"
            and retention.get("limits") == LIMITS and retention.get("qualification_passed") is False,
            "Missing or changed result retention contract")
    require(files["retention.json"] == expected["retention"] and len(files) == expected["regular_files"],
            "Returned retention identity or file count differs")
    selected = {name: entry for name, entry in files.items() if name != "retention.json"}
    require(selected == retention["files"]
            and retention["payload_bytes"] == sum(item["bytes"] for item in selected.values())
            and retention["metadata_bytes"] == sum(item["bytes"] for name, item in selected.items()
                                                   if name not in MODEL_RESULTS),
            "Returned files do not reproduce the retention inventory")
    return {"archive": identity(path), "files": files, "payload_bytes": payload,
            "regular_files": len(files), "qualification_passed": False}


def numerical_readback(directory):
    """Recompute reported comparisons from non-pickle arrays, including failures."""
    import numpy as np

    directory = Path(directory)
    def array(path):
        require(path.stat().st_size <= 8 * 1024**2, "Numerical array exceeds its bound")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        require(values.ndim == 2 and 1 <= values.shape[0] <= 3 and 1 <= values.shape[1] <= 131072
                and values.dtype in (np.dtype("float32"), np.dtype("float64")),
                "Numerical array shape or dtype differs")
        return values

    result = {"merge": None, "export": {}, "numerical_gates_passed": False,
              "qualification_passed": False}
    merge_path = directory / "workload/merge/report.json"
    if not merge_path.exists():
        result["unavailable"] = "No completed merge report was returned"
        return result
    merge = load(merge_path)
    require(merge.get("maximum_merge_logit_error") == 0.0002, "Merge tolerance changed")
    arrays = {}
    for label in ("base", "adapter", "merged"):
        name = label + "-reference-logits.npy"
        path = merge_path.parent / name
        if not path.exists():
            result["unavailable"] = "Incomplete merge reference arrays"
            return result
        require(identity(path) == merge["references"][name], "Merge reference identity differs")
        arrays[label] = array(path)
    comparison = check_merge(arrays["adapter"], arrays["merged"], maximum_error=0.0002)
    require(comparison == merge["merge"]["comparison"], "Merge comparison does not reproduce")
    result["merge"] = comparison
    report_path = directory / "workload/export/report.json"
    if not report_path.exists():
        result["unavailable"] = "No export report was returned"
        return result
    report = load(report_path)
    require(report["limits"]["f32_maximum_log_probability_error"] == 0.002
            and report["limits"]["quantized_maximum_mean_kl"] == 0.10, "Export tolerances changed")
    failures = []
    for label, stored in report["comparisons"].items():
        require(label in ("stock-f32", "trained-f32", "stock-q4", "trained-q4"),
                "Unexpected numerical comparison")
        observed = array(report_path.parent / (label + "-log-probabilities.npy"))
        comparison = compare_native_probabilities(arrays["base" if label.startswith("stock") else "merged"],
                                                  observed)
        # Independent hosts can differ in their last floating-point digits.
        for key in ("maximum_absolute_log_probability_error", "mean_kl_divergence",
                    "maximum_normalized_log_probability_error"):
            require(abs(comparison[key] - stored[key]) <= 1e-10, "Numerical comparison does not reproduce")
        require(comparison["top_one_agreement"] == stored["top_one_agreement"]
                and np.allclose(comparison["per_row_kl_divergence"], stored["per_row_kl_divergence"],
                                atol=1e-10, rtol=0), "Numerical rows do not reproduce")
        if label.endswith("f32") and comparison["maximum_absolute_log_probability_error"] > 0.002:
            failures.append(label + ": F32 raw log-probability error")
        if label.endswith("q4") and comparison["mean_kl_divergence"] > 0.10:
            failures.append(label + ": Q4 mean divergence")
        result["export"][label] = comparison
    require(failures == report["numerical_failures"], "Numerical failure inventory differs")
    result["numerical_gates_passed"] = len(result["export"]) == 4 and not failures
    result["reported_export_passed"] = report["passed"]
    result["reported_export_failure"] = report["failure"]
    require(not report["passed"] or result["numerical_gates_passed"]
            and report["failure"] is None and report["export_pair_verified"] is True,
            "Claimed export success contradicts numerical evidence")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extract", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    expected = load(args.identity)
    result = read_archive(args.archive, expected)
    if args.extract:
        require(shutil.disk_usage(args.extract.parent).free >= result["payload_bytes"] + 1024**3,
                "Insufficient disk space for extraction and a 1 GiB reserve")
        require(read_archive(args.archive, expected, destination=args.extract) == result,
                "Extracted result differs from its inspected archive")
        result["numerical_readback"] = numerical_readback(args.extract)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
