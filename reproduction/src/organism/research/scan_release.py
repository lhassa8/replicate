"""Report credential-pattern findings without printing matching secret values."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


PATTERNS = {
    "private_key_material": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----(?:\r?\n|\\n)"
        rb"[A-Za-z0-9+/=\\\r\n]{64,}"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "presigned_signature": re.compile(rb"X-Amz-Signature=[a-fA-F0-9]{64}"),
    "serialized_secret_key": re.compile(
        rb'["\'](?:SecretAccessKey|aws_secret_access_key)["\']\s*:\s*["\'][A-Za-z0-9+/]{30,}'),
    "serialized_session_token": re.compile(
        rb'["\'](?:SessionToken|aws_session_token)["\']\s*:\s*["\'][A-Za-z0-9+/=]{40,}'),
    "bearer_value": re.compile(rb"Bearer [A-Za-z0-9._~+/=-]{32,}"),
    "observation_secret_value": re.compile(
        rb'["\']observation_secret["\']\s*:\s*["\'][a-fA-F0-9]{32,}'),
}


def scan(root, assembly, output):
    files = [(str(p.relative_to(root)), p) for p in root.rglob("*") if p.is_file()]
    if assembly:
        files += [(name, Path(record["path"]))
                  for name, record in json.loads(assembly.read_text()).items()]
    findings, skipped, checked = [], [], 0

    def inspect(name, data):
        nonlocal checked
        checked += 1
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                findings.append({"file": name, "pattern": label,
                                 "file_sha256": hashlib.sha256(data).hexdigest()})

    for name, path in sorted(files):
        if path.suffix == ".gguf":
            skipped.append({"file": name, "reason": "Quantized model weights; identity checked separately"})
        elif path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.isfile():
                        if member.size > 128 * 1024**2:
                            skipped.append({"file": name + "::" + member.name,
                                            "reason": "Archive member over scan bound"})
                        else:
                            inspect(name + "::" + member.name, archive.extractfile(member).read())
        elif path.stat().st_size <= 128 * 1024**2:
            inspect(name, path.read_bytes())
        else:
            skipped.append({"file": name, "reason": "File over scan bound"})
    result = {"schema": "replicate-reproduction-credential-pattern-scan-v1",
              "files_or_archive_members_scanned": checked, "patterns": sorted(PATTERNS),
              "findings": findings, "skipped": skipped,
              "scope": "Finite credential-pattern check; not a general sensitive-data guarantee",
              "matching_values_disclosed": False}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scanned": checked, "findings": findings, "skipped": len(skipped)}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scan(args.package.resolve(), args.assembly.resolve() if args.assembly else None,
         args.output.resolve())
