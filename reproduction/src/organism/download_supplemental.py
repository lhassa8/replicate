"""Download exact Ubuntu inputs declared in retained publisher metadata."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from .ubuntu_supplemental import ORIGINS, decompress_index, fetch, index_identity, release_body, save, sha, stanzas


def declared_package(metadata, architecture, pocket, package, version):
    location = Path(metadata) / architecture / pocket
    receipt = json.loads((location / "receipt.json").read_bytes())
    raw_release = (location / "InRelease").read_bytes()
    raw_index = (location / "Packages.xz").read_bytes()
    if (sha(raw_release) != receipt["release"]["sha256"] or len(raw_release) != receipt["release"]["bytes"]
            or sha(raw_index) != receipt["index"]["sha256"] or len(raw_index) != receipt["index"]["bytes"]):
        raise ValueError("Retained Ubuntu metadata has changed")
    fields = release_body(raw_release)
    if fields["Suite"] != pocket:
        raise ValueError("Retained metadata belongs to another Ubuntu pocket")
    length, identity = index_identity(fields, "main/binary-" + architecture + "/Packages.xz")
    if length != len(raw_index) or identity != sha(raw_index):
        raise ValueError("Ubuntu package index no longer matches its release")
    matches = [record for record in stanzas(decompress_index(raw_index))
               if record.get("Package") == package and record.get("Version") == version
               and record.get("Architecture") == architecture]
    if len(matches) != 1:
        raise ValueError("Pinned package is not uniquely declared in its architecture index")
    value = matches[0]
    return {key: value[key] for key in ("Package", "Version", "Architecture", "Filename", "SHA256", "Size")}


def download(root, pins, output):
    root, pins, output = Path(root).resolve(), Path(pins).resolve(), Path(output).absolute()
    request = json.loads(pins.read_bytes())
    if (request["schema"] != "organism-ubuntu-supplemental-pins-v1"
            or set(request["architectures"]) != set(ORIGINS)):
        raise ValueError("Unexpected supplemental pin declaration")
    metadata = root / request["metadata"]
    tasks = []
    for architecture in request["architectures"]:
        for package, wanted in request["packages"].items():
            declaration = declared_package(metadata, architecture, wanted["pocket"], package, wanted["version"])
            if not 0 < int(declaration["Size"]) <= 16 * 1024**2:
                raise ValueError("Pinned package exceeds its download bound")
            path = declaration["Filename"]
            if not path.startswith("pool/main/") or any(part in ("", ".", "..") for part in path.split("/")):
                raise ValueError("Pinned package has an unsafe publisher path")
            tasks.append((architecture, wanted["pocket"], declaration))
    output.mkdir(parents=True, exist_ok=False)
    for architecture in request["architectures"]:
        (output / architecture).mkdir()
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(fetch, ORIGINS[architecture] + "/" + value["Filename"],
                               output / architecture / Path(value["Filename"]).name,
                               maximum=16 * 1024**2, expected_bytes=int(value["Size"]), expected_sha256=value["SHA256"])
                   for architecture, _, value in tasks]
        packages, failures = [], []
        for (architecture, pocket, declaration), future in zip(tasks, pending):
            try:
                receipt = future.result()
                packages.append({"architecture": architecture, "pocket": pocket, "declaration": declaration,
                                 "download": receipt, "relative_path": architecture + "/" + Path(declaration["Filename"]).name})
            except Exception as exc:
                failures.append({"architecture": architecture, "package": declaration["Package"],
                                 "failure": type(exc).__name__ + ": " + str(exc)})
    report = {"scope": "Pinned official Ubuntu packages, not installed or executed",
              "recorded_at": datetime.now(timezone.utc).isoformat(), "pins_sha256": sha(pins.read_bytes()),
              "pins": request, "packages": packages, "failures": failures,
              "source_sha256": sha(Path(__file__).read_bytes()),
              "metadata_signatures_pending_native_keyring_verification": True,
              "native_execution_verified": False, "qualification_passed": False}
    save(output / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = download(Path(__file__).resolve().parent, args.pins, args.output)
    print(json.dumps({"packages": len(report["packages"]), "failures": report["failures"]}))
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
