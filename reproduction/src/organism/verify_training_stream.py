"""Verify pinned native source bytes in transit without retaining large weights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

from .training_inputs import load, require, sha256, validate_recipe, verified_path
from .training_source import inspect_headers, inspect_recipe_source


FILES = {"config.json", "model.safetensors.index.json",
         "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
CHUNK_BYTES = 1024 * 1024
MAXIMUM_SECONDS = 1800


class HttpsRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        require(urllib.parse.urlsplit(newurl).scheme == "https", "Source redirected outside HTTPS")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def stream_file(opener, entry, expected_prefix, deadline, state, persist):
    """Retain only the known bounded prefix; read and hash the entire response."""
    require(0 < len(expected_prefix) <= 1024 * 1024
            and type(entry["bytes"]) is int and len(expected_prefix) <= entry["bytes"] <= 6 * 1024**3,
            "Unexpected file or prefix bound")
    require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid source hash")
    request = urllib.request.Request(entry["source_url"], headers={"Accept-Encoding": "identity"})
    digest, prefix, total = hashlib.sha256(), bytearray(), 0
    started, last_saved = time.monotonic(), time.monotonic()
    state.update(status="streaming", received_bytes=0, full_payload_verified=False)
    persist()
    try:
        require(started < deadline, "Source verification deadline expired")
        with opener.open(request, timeout=30) as response:
            require(response.status == 200, "Source did not return a complete HTTP 200 response")
            require(response.headers.get("Content-Encoding", "identity") == "identity",
                    "Encoded transfer would change the pinned byte representation")
            length = response.headers.get("Content-Length")
            require(length is not None and int(length) == entry["bytes"], "Declared source length differs")
            state["final_source_host"] = urllib.parse.urlsplit(response.geturl()).hostname
            while True:
                require(time.monotonic() < deadline, "Source verification deadline expired")
                block = response.read(min(CHUNK_BYTES, entry["bytes"] - total + 1))
                if not block:
                    break
                total += len(block)
                state["received_bytes"] = total
                require(total <= entry["bytes"], "Source exceeded its pinned length")
                digest.update(block)
                if len(prefix) < len(expected_prefix):
                    prefix.extend(block[:len(expected_prefix) - len(prefix)])
                    require(bytes(prefix) == expected_prefix[:len(prefix)], "Source prefix differs from pinned metadata")
                now = time.monotonic()
                if now - last_saved >= 5:
                    state["seconds"] = round(now - started, 3)
                    persist()
                    last_saved = now
        require(total == entry["bytes"] and digest.hexdigest() == entry["sha256"],
                "Full source payload does not match its pinned size and hash")
        require(time.monotonic() < deadline, "Source verification deadline expired")
        state.update(status="verified", received_bytes=total, sha256=digest.hexdigest(),
                     retained_prefix_bytes=len(prefix), full_payload_verified=True)
        return bytes(prefix)
    finally:
        state["seconds"] = round(time.monotonic() - started, 3)
        persist()


def run(root, output, *, maximum_seconds=MAXIMUM_SECONDS, opener=None):
    root, output = Path(root).resolve(), Path(output)
    require(type(maximum_seconds) is int and 1 <= maximum_seconds <= MAXIMUM_SECONDS,
            "Streaming deadline must be one to 1800 seconds")
    recipe_path = root / "organism/training-recipe.draft.json"
    recipe = load(recipe_path)
    validate_recipe(recipe)
    paths = {name: verified_path(root, entry) for name, entry in recipe["inputs"].items()}
    metadata = load(paths["base_metadata"])
    require(metadata["repository"] == recipe["base"]["repository"]
            and metadata["revision"] == recipe["base"]["revision"]
            and set(metadata["files"]) == FILES, "Native source differs from the bound recipe")
    source_check = inspect_recipe_source(paths, recipe["base"]["expected_parameters"])
    for name, entry in metadata["files"].items():
        expected = ("https://huggingface.co/" + recipe["base"]["repository"] + "/resolve/"
                    + recipe["base"]["revision"] + "/" + name)
        require(entry["source_url"] == expected, "Download URL differs from the pinned revision")
    output.mkdir(parents=True, exist_ok=False)
    retained = output / "prefixes"
    retained.mkdir()
    started = time.monotonic()
    report = {
        "schema": "organism-native-source-stream-check-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "streaming", "recipe_sha256": sha256(recipe_path),
        "metadata_sha256": sha256(paths["base_metadata"]), "repository": metadata["repository"],
        "revision": metadata["revision"], "maximum_seconds": maximum_seconds,
        "maximum_chunk_bytes": CHUNK_BYTES, "files": {}, "failure": None,
        "full_source_payload_verified_in_transit": False, "large_weight_files_retained": False,
        "native_model_loaded": False, "cuda_verified": False, "optimizer_steps": 0,
        "qualification_passed": False,
    }

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        temporary = output / ".report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "report.json")

    persist()
    try:
        opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), HttpsRedirects())
        for name, entry in metadata["files"].items():
            if name.endswith(".safetensors"):
                prefix_path, saved_name = paths["base_header/" + name], name + ".header"
            else:
                prefix_path = paths["base_config" if name == "config.json" else "base_index"]
                saved_name = name
            state = {"source_url": entry["source_url"], "expected_bytes": entry["bytes"],
                     "expected_sha256": entry["sha256"], "full_payload_verified": False}
            report["files"][name] = state
            prefix = stream_file(opener, entry, prefix_path.read_bytes(), started + maximum_seconds, state, persist)
            saved = retained / saved_name
            saved.write_bytes(prefix)
            state["retained_prefix"] = {"path": str(saved.relative_to(output)), "bytes": len(prefix), "sha256": sha256(saved)}
        headers = {name: retained / (name + ".header") for name in FILES if name.endswith(".safetensors")}
        report["streamed_header_check"] = inspect_headers(
            metadata, load(paths["loader_index"]), headers, load(retained / "config.json"),
            recipe["base"]["expected_parameters"])
        report["streamed_header_check"]["scope"] = "Headers captured from fully hashed streams; large weight files were not retained"
        report["upstream_index_consistent"] = source_check["upstream_index_consistent"]
        report["upstream_index_placement_mismatches"] = source_check["upstream_index_placement_mismatches"]
        report["derived_loader_index_sha256"] = sha256(paths["loader_index"])
        report["full_source_payload_verified_in_transit"] = all(
            state["full_payload_verified"] for state in report["files"].values())
        report["status"] = "verified_in_transit_not_loaded"
        report["total_bytes_verified"] = sum(state["received_bytes"] for state in report["files"].values())
        report["retained_prefix_bytes"] = sum(state["retained_prefix"]["bytes"] for state in report["files"].values())
        report["limitations"] = [
            "The GPU worker must download and verify its own complete files before loading.",
            "Stream integrity and tensor layout do not establish numerical model behavior or device compatibility.",
            "No model loading, optimization, cloud resource creation or qualification occurred.",
        ]
    except BaseException as exc:
        report["status"] = "failed"
        # CDN redirect URLs can carry temporary credentials; do not retain them.
        message = re.sub(r"https?://\S+", "<url>", str(exc))
        report["failure"] = {"type": type(exc).__name__, "message": message[:1000]}
        for state in report["files"].values():
            if state.get("status") == "streaming":
                state["status"] = "failed"
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        persist()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=int, default=MAXIMUM_SECONDS)
    args = parser.parse_args()
    result = run(Path(__file__).resolve().parents[1], args.output, maximum_seconds=args.maximum_seconds)
    print(json.dumps(result, indent=2))
