"""Acquire and inspect Ubuntu library inputs without installing or executing them.

Official HTTPS metadata and its SHA-256 chain are retained. This Mac acquisition
does not verify OpenPGP signatures; native preparation must verify the retained
InRelease files against the image's Ubuntu archive keyring before use.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import lzma
from pathlib import Path
import re
import urllib.parse
import urllib.request


ORIGINS = {
    "amd64": "https://archive.ubuntu.com/ubuntu",
    "arm64": "https://ports.ubuntu.com/ubuntu-ports",
}
POCKETS = ("noble", "noble-updates", "noble-security")
PACKAGES = frozenset(("libgcc-s1", "libstdc++6", "libgomp1", "libssl3t64", "libcrypt1",
                      "gcc-14-base", "gcc-13-base", "ubuntu-keyring"))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2)
        output.write("\n")


def fetch(url, path, *, maximum, expected_bytes=None, expected_sha256=None):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in ("archive.ubuntu.com", "ports.ubuntu.com"):
        raise ValueError("Only the declared official HTTPS origins are allowed")
    with urllib.request.urlopen(url, timeout=45) as response:
        if urllib.parse.urlparse(response.url).netloc != parsed.netloc:
            raise ValueError("Ubuntu download redirected outside its origin")
        size = response.headers.get("Content-Length")
        if size and (int(size) > maximum or expected_bytes is not None and int(size) != expected_bytes):
            raise ValueError("Ubuntu download length differs from its bound")
        raw = response.read(maximum + 1)
    if len(raw) > maximum or expected_bytes is not None and len(raw) != expected_bytes:
        raise ValueError("Ubuntu download exceeds its declared length")
    if expected_sha256 is not None and sha(raw) != expected_sha256:
        raise ValueError("Ubuntu download differs from its metadata hash")
    with Path(path).open("xb") as output:
        output.write(raw)
    return {"url": url, "bytes": len(raw), "sha256": sha(raw)}


def stanzas(text):
    """Read Debian control fields while rejecting duplicate or orphan fields."""
    record, field = {}, None
    for line in text.splitlines() + [""]:
        if not line:
            if record:
                yield record
            record, field = {}, None
        elif line[0].isspace():
            if field is None:
                raise ValueError("Orphan Debian field continuation")
            record[field] += "\n" + line[1:]
        else:
            field, colon, value = line.partition(":")
            if not colon or field in record or not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", field):
                raise ValueError("Malformed or duplicate Debian field")
            record[field] = value.lstrip()


def release_body(raw):
    text = raw.decode()
    start = text.index("\n\n", text.index("-----BEGIN PGP SIGNED MESSAGE-----")) + 2
    end = text.index("-----BEGIN PGP SIGNATURE-----", start)
    # OpenPGP cleartext dash escaping is removed before parsing the fields.
    cleartext = "\n".join(line[2:] if line.startswith("- ") else line for line in text[start:end].splitlines())
    records = list(stanzas(cleartext))
    if len(records) != 1 or records[0].get("Origin") != "Ubuntu" or records[0].get("Codename") != "noble":
        raise ValueError("Unexpected Ubuntu release identity")
    return records[0]


def index_identity(release, relative):
    matches = []
    for line in release["SHA256"].splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == relative:
            matches.append(fields)
    if len(matches) != 1 or not re.fullmatch("[0-9a-f]{64}", matches[0][0]):
        raise ValueError("Package index lacks a unique SHA-256 declaration")
    return int(matches[0][1]), matches[0][0]


def decompress_index(raw):
    # Ubuntu base indexes can append an empty second XZ stream. Support bounded
    # concatenation and four-byte XZ padding without discarding arbitrary tails.
    pieces, total = [], 0
    for _ in range(8):
        decoder = lzma.LZMADecompressor(memlimit=128 * 1024**2)
        value = decoder.decompress(raw, max_length=128 * 1024**2 - total + 1)
        total += len(value)
        if not decoder.eof or total > 128 * 1024**2:
            raise ValueError("Package index exceeds its decompression bound")
        pieces.append(value)
        raw = decoder.unused_data
        padding = len(raw) - len(raw.lstrip(b"\0"))
        if padding % 4:
            raise ValueError("Malformed XZ stream padding")
        raw = raw[padding:]
        if not raw:
            return b"".join(pieces).decode()
    raise ValueError("Package index contains too many XZ streams")


def acquire_one(root, architecture, pocket):
    target = root / architecture / pocket
    target.mkdir(parents=True, exist_ok=False)
    base = ORIGINS[architecture]
    release = fetch(base + "/dists/" + pocket + "/InRelease", target / "InRelease", maximum=1024**2)
    fields = release_body((target / "InRelease").read_bytes())
    if fields.get("Suite") != pocket or architecture not in fields.get("Architectures", "").split():
        raise ValueError("Release does not describe the requested pocket and architecture")
    relative = "main/binary-" + architecture + "/Packages.xz"
    length, identity = index_identity(fields, relative)
    if not 0 < length <= 16 * 1024**2:
        raise ValueError("Package index exceeds its transfer bound")
    index = fetch(base + "/dists/" + pocket + "/" + relative, target / "Packages.xz",
                  maximum=16 * 1024**2, expected_bytes=length, expected_sha256=identity)
    candidates = []
    for record in stanzas(decompress_index((target / "Packages.xz").read_bytes())):
        if record.get("Package") not in PACKAGES or record.get("Architecture") not in (architecture, "all"):
            continue
        name = record["Filename"]
        if (not name.startswith("pool/main/") or any(part in ("", ".", "..") for part in name.split("/"))
                or not re.fullmatch(r"[A-Za-z0-9_./+~%-]+", name)):
            raise ValueError("Noncanonical Ubuntu package path")
        candidate = {key: record[key] for key in ("Package", "Version", "Architecture", "Filename", "SHA256", "Size")}
        candidate.update({key: record.get(key) for key in ("Source", "Depends", "Description")})
        candidates.append(candidate)
    result = {
        "architecture": architecture, "pocket": pocket, "release": release, "index": index,
        "release_date": fields.get("Date"), "valid_until": fields.get("Valid-Until"),
        "candidates": candidates, "openpgp_verified": False,
    }
    save(target / "receipt.json", result)
    return result


def acquire(root):
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=False)
    requests = [(root, architecture, pocket) for architecture in ORIGINS for pocket in POCKETS]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(acquire_one, *request) for request in requests]
        results, failures = [], []
        for request, future in zip(requests, pending):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"architecture": request[1], "pocket": request[2],
                                 "failure": type(exc).__name__ + ": " + str(exc)})
    result = {
        "scope": "Official HTTPS Ubuntu inputs with SHA-256 chain; OpenPGP verification pending native keyring",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(Path(__file__).read_bytes()), "sources": results, "failures": failures,
        "openpgp_verified": False, "native_execution_verified": False, "qualification_passed": False,
    }
    save(root / "receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = acquire(args.output)
    print(json.dumps({"sources": len(result["sources"]), "failures": result["failures"],
                      "openpgp_verified": False}))
    return bool(result["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
