"""Reconstruct the original signed release from packaged, verified components."""

import argparse
import base64
import json
from pathlib import Path
import shutil

from organism.capsule import verify_capsule_directory, verify_manifest
from organism.research.build_reproduction_package import identity


def reconstruct(package, output=None):
    assets = package / "assets"
    raw = (assets / "original-release/manifest.json").read_bytes()
    manifest = json.loads(raw)
    declaration = json.loads((assets / "original-release/release-trust.json").read_text())
    if (declaration["schema"] != "organism-development-release-trust-v1"
            or declaration["release_id"] != manifest["release_id"]
            or declaration["require_trained"] is not True):
        raise ValueError("Original public trust declaration differs")
    trust = {k: base64.b64decode(v, validate=True)
             for k, v in declaration["public_keys"].items()}
    verify_manifest(raw, trust)
    body = manifest["body"]
    sources = {}
    for name, expected in body["files"].items():
        source = (assets / "models/trained.gguf" if name == body["model"]["path"]
                  else assets / "original-release/agent.tar.gz" if name == body["agent"]
                  else assets / "shared" / name)
        if identity(source) != expected:
            raise ValueError("Original release component differs: " + name)
        sources[name] = source
    if output:
        output.mkdir(parents=True, exist_ok=False)
        (output / ".manifest.json").write_bytes(raw)
        for name, source in sources.items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        verify_capsule_directory(output, raw, trust)
    result = {"release_id": manifest["release_id"], "components_verified": len(sources),
              "payload_bytes": sum(row["bytes"] for row in body["files"].values()),
              "capsule_reconstructed": output is not None, "native_execution_performed": False}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify-inputs-only", action="store_true")
    args = parser.parse_args()
    reconstruct(args.package.resolve(), args.output.resolve() if args.output else None)
