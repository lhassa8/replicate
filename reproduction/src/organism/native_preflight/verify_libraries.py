"""Verify carried publisher signatures and native library resolution on Linux."""

import json
from pathlib import Path
import re
import subprocess
import tempfile

from .common import check_file, digest, write_json
from .libraries import BASE, elf_files, inside, resolve
from ..ubuntu_supplemental import decompress_index, index_identity, release_body, stanzas


KEYRING = Path("/usr/share/keyrings/ubuntu-archive-keyring.gpg")


def verified_release(path, keyring=KEYRING):
    """Use the image's trusted keyring, never keys supplied by the package."""
    path, keyring = Path(path), Path(keyring)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("Invalid retained Ubuntu signed metadata")
    if keyring.is_symlink() or not keyring.is_file() or keyring.stat().st_size > 1024**2:
        raise ValueError("Trusted image keyring is missing or invalid")
    with tempfile.TemporaryDirectory(prefix="organism-gpgv-") as temporary:
        output = Path(temporary) / "authenticated-release"
        result = subprocess.run(
            ["/usr/bin/gpgv", "--homedir", temporary, "--keyring", str(keyring),
             "--status-fd", "1", "--output", str(output), str(path)],
            capture_output=True, text=True, timeout=20,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        valid = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[:2] == ["[GNUPG:]", "VALIDSIG"]:
                if not re.fullmatch(r"[A-Fa-f0-9]{40,64}", fields[2]):
                    raise ValueError("GnuPG returned a malformed signer fingerprint")
                valid.append(fields[2].upper())
            if len(fields) >= 2 and fields[0] == "[GNUPG:]" and fields[1] in {
                    "BADSIG", "ERRSIG", "EXPSIG", "EXPKEYSIG", "REVKEYSIG", "NO_PUBKEY", "FAILURE"}:
                raise ValueError("Ubuntu publisher signature was rejected")
        if (result.returncode != 0 or not valid or not output.is_file()
                or output.stat().st_size > 1024**2):
            raise ValueError("Ubuntu publisher signature could not be verified")
        records = list(stanzas(output.read_text()))
        if len(records) != 1 or records[0] != release_body(path.read_bytes()):
            raise ValueError("Authenticated release differs from the retained metadata parser")
    return records[0], {"keyring_sha256": digest(keyring), "valid_signer_fingerprints": valid,
                        "gpgv_return_code": result.returncode}


def verify_metadata(supplemental, architecture, *, keyring=KEYRING):
    supplemental = Path(supplemental)
    wanted_arch = {"x86_64": "amd64", "arm64": "arm64"}[architecture]
    receipt = json.loads((supplemental / "library-resolution.json").read_bytes())
    if receipt["architecture"] != wanted_arch:
        raise ValueError("Carried metadata belongs to another native architecture")
    packages = receipt["packages"]
    expected_packages = {"base-files", "gcc-14-base", "libgcc-s1", "libgomp1", "libstdc++6", "libssl3t64", "libcrypt1"}
    if (len(packages) != 7 or {entry["declaration"]["Package"] for entry in packages} != expected_packages
            or any(entry["architecture"] != wanted_arch or entry["declaration"]["Architecture"] != wanted_arch
                   or entry["pocket"] not in ("noble", "noble-updates", "noble-security") for entry in packages)):
        raise ValueError("Carried package declarations are incomplete or duplicated")
    libraries = receipt["carried_libraries"]
    required = {"libgcc_s.so.1", "libstdc++.so.6", "libgomp.so.1", "libssl.so.3", "libcrypto.so.3", "libcrypt.so.1"}
    if not isinstance(libraries, dict) or not required <= set(libraries) or len(libraries) > 128:
        raise ValueError("Carried native library inventory is incomplete")
    results = {}
    for pocket in sorted({entry["pocket"] for entry in packages}):
        location = supplemental / "publisher-metadata" / pocket
        source = json.loads((location / "receipt.json").read_bytes())
        check_file(location / "InRelease", source["release"])
        check_file(location / "Packages.xz", source["index"])
        fields, verification = verified_release(location / "InRelease", keyring)
        if fields.get("Suite") != pocket or wanted_arch not in fields.get("Architectures", "").split():
            raise ValueError("Authenticated release has another suite or architecture")
        count, identity = index_identity(fields, "main/binary-" + wanted_arch + "/Packages.xz")
        check_file(location / "Packages.xz", {"bytes": count, "sha256": identity})
        index = list(stanzas(decompress_index((location / "Packages.xz").read_bytes())))
        for entry in packages:
            if entry["pocket"] != pocket or entry["architecture"] != wanted_arch:
                continue
            declared = entry["declaration"]
            matches = [record for record in index if all(record.get(name) == declared[name]
                       for name in ("Package", "Version", "Architecture", "Filename", "SHA256", "Size"))]
            if len(matches) != 1:
                raise ValueError("Carried package is not bound to the authenticated index")
        results[pocket] = {**verification, "release_sha256": digest(location / "InRelease"),
                           "index_sha256": identity, "release_date": fields.get("Date")}
    for name, entry in libraries.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Unsafe carried library identity")
        check_file(supplemental / "lib" / name, entry)
        origins = [item for item in packages if item["declaration"]["Package"] == entry["package"]
                   and item["declaration"]["Version"] == entry["package_version"]
                   and item["declaration"]["SHA256"] == entry["publisher_package_sha256"]]
        if len(origins) != 1:
            raise ValueError("Carried library lacks its publisher package identity")
    return {"architecture": wanted_arch, "releases": results, "publisher_signatures_verified": True,
            "carried_library_hashes_verified": True}


def verify_carried(root, output, architecture):
    root, output = Path(root), Path(output)
    supplemental = root / "supplemental"
    signatures = verify_metadata(supplemental, architecture)
    roots = [root / "python", root / "tools", root / "site-packages", supplemental / "lib"]
    search = [supplemental / "lib", root / "python/lib", root / "tools/llama.cpp/b9850/llama-b9850"]
    objects, substrate = list(elf_files(roots)), {}
    for binary in objects:
        for name, path in resolve(binary, search):
            if name in BASE:
                substrate[name] = {"path": str(path), "sha256": digest(path)}
            elif not inside(path, roots):
                raise ValueError("Native loading would require an uncarried host library: " + name)
    report = {"scope": "Native ldd resolution and Ubuntu publisher-signature checks; dynamic execution is a separate test",
              "elf_objects_checked": len(objects), "base_os_libraries": substrate,
              "search_paths": list(map(str, search)), "publisher_verification": signatures,
              "missing_libraries_copied_from_host": False, "qualification_passed": False}
    write_json(output / "library-resolution.json", report)
    return search
