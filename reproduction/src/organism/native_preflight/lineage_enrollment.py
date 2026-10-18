"""Fresh owner enrollment for three execution slots and one seed-transfer key.

The fourth identity is an owner transport principal. No receiver or application
is installed for it. Each of the three actual device slots gets only its own
private key; the seed also receives the separate owner's seed-transfer key.
"""

import base64
import gzip
import io
import json
import os
from pathlib import Path
import stat
import tarfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ..capsule import canonical
from ..discovery_book import DiscoveryBook, sign_discovery_book
from ..enrollment import Enrollment, key_id, public_bytes, sign_enrollment
from ..peer_identity import context, device_certificate, owner_certificate, write_identity
from ..training_inputs import require
from . import lineage_network as network
from .common import digest
from .trained_inputs import validate_catalog


MAX_BUNDLE = 64 * 1024


def file_identity(path):
    return {"bytes": Path(path).stat().st_size, "sha256": digest(path)}


def private_write(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def pack(directory, target):
    directory, target = Path(directory), Path(target)
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    require(len(files) <= 16 and sum(path.stat().st_size for path in files) <= MAX_BUNDLE,
            "Owner identity bundle exceeds its bound")
    with target.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        os.fchmod(raw.fileno(), 0o600)
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for path in files:
                entry = path.lstat()
                require(stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1
                        and not entry.st_mode & 0o077, "Owner bundle contains a link or public private file")
                payload = path.read_bytes()
                member = tarfile.TarInfo(str(path.relative_to(directory)))
                member.size, member.mode, member.mtime = len(payload), 0o600, 0
                archive.addfile(member, io.BytesIO(payload))
    require(target.stat().st_size <= MAX_BUNDLE, "Compressed owner bundle exceeds its bound")
    return file_identity(target)


def extract_bundle(archive_path, target):
    archive_path, target = Path(archive_path), Path(target)
    require(0 < archive_path.stat().st_size <= MAX_BUNDLE, "Owner archive exceeds its bound")
    require(not target.exists() and not target.is_symlink(), "Owner extraction directory must be new")
    with tarfile.open(archive_path, "r:gz") as archive:
        entries = archive.getmembers()
        require(0 < len(entries) <= 16 and sum(entry.size for entry in entries) <= MAX_BUNDLE,
                "Expanded owner bundle exceeds its bound")
        seen = set()
        for entry in entries:
            name = Path(entry.name)
            require(entry.isfile() and 0 <= entry.size <= MAX_BUNDLE
                    and not name.is_absolute() and ".." not in name.parts
                    and "\\" not in entry.name and str(name) == entry.name
                    and len(name.parts) <= 2 and entry.name not in seen,
                    "Unsafe or repeated owner bundle entry")
            seen.add(entry.name)
        target.mkdir(mode=0o700)
        for entry in entries:
            path = target / entry.name
            if path.parent != target:
                path.parent.mkdir(mode=0o700, exist_ok=True)
            with archive.extractfile(entry) as stream:
                raw = stream.read(entry.size + 1)
            require(len(raw) == entry.size, "Truncated owner bundle")
            private_write(path, raw)


def prepare(catalog, output):
    """Prepare identity inputs only; no host, receiver, or descendant is started."""
    validate_catalog(catalog)
    output = Path(output).absolute()
    workspace = Path(__file__).resolve().parents[2]
    require(not output.resolve().is_relative_to(workspace),
            "Ephemeral private enrollment keys must stay outside the repository")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    output.chmod(0o700)
    manifest = json.loads(base64.b64decode(catalog["manifest_base64"], validate=True))
    owner = Ed25519PrivateKey.generate()
    owner_public = public_bytes(owner.public_key())
    slots = (*network.SLOTS, "bootstrap")
    keys = {slot: Ed25519PrivateKey.generate() for slot in slots}
    ids = {slot: key_id(public_bytes(key.public_key())) for slot, key in keys.items()}
    require(len(set(ids.values())) == 4, "Fresh owner identities unexpectedly collided")
    body = {
        "schema": "organism-enrollment-v1", "experiment_id": manifest["body"]["experiment_id"],
        "epoch": 1, "devices": {
            ids[slot]: {"public_key": base64.b64encode(public_bytes(keys[slot].public_key())).decode(),
                        "platform": "linux-x86_64"} for slot in slots},
        "release_keys": catalog["release_keys"], "require_trained": True,
        "max_payload_bytes": 3 * 1024**3, "birth_seconds": 1200,
        "max_births_per_device": 1, "max_process_starts": 1,
    }
    policy = Enrollment(sign_enrollment(body, owner), owner_public)
    addresses = {**network.ADDRESSES, "bootstrap": network.BOOTSTRAP_ADDRESS}
    book = DiscoveryBook(sign_discovery_book({
        "schema": "organism-discovery-book-v1", "policy_id": policy.policy_id, "epoch": 1,
        "lab_cidrs": [network.LAB],
        "endpoints": {ids[slot]: {"address": addresses[slot], "port": 5443} for slot in slots},
    }, owner, policy), policy, owner_public)
    certificate = owner_certificate(owner, days=1)
    bundles = {}
    for slot in network.SLOTS:
        node = output / slot
        node.mkdir(mode=0o700)
        for name, raw in (
            ("enrollment.json", policy.signed), ("discovery.json", book.signed),
            ("node.json", canonical({
                "schema": "organism-lineage-owner-node-v1", "slot": slot, "device_id": ids[slot],
                "release_id": catalog["release_id"], "policy_id": policy.policy_id,
                "owner_public": base64.b64encode(owner_public).decode(), "device_ids": ids,
                "execution_slots": list(network.SLOTS), "bootstrap_has_execution_slot": False,
                "underlay_address": network.UNDERLAY[slot], "bind_address": addresses[slot],
                "initial_owner_mode": "stopped", "qualification_passed": False,
            })),
        ):
            private_write(node / name, raw)
        write_identity(node / "identity", keys[slot],
                       device_certificate(owner, certificate, keys[slot].public_key(), days=1), certificate)
        if slot == "seed":
            write_identity(node / "bootstrap", keys["bootstrap"],
                           device_certificate(owner, certificate, keys["bootstrap"].public_key(), days=1),
                           certificate)
        verify(node, slot, catalog)
        target = output / (slot + ".tar.gz")
        bundles[slot] = {"path": str(target), **pack(node, target)}
    public_directory = output / "public"
    public_directory.mkdir(mode=0o700)
    for name, source in {
        "enrollment.json": output / "seed/enrollment.json",
        "discovery.json": output / "seed/discovery.json",
        "owner-certificate.pem": output / "seed/identity/owner.pem",
        **{slot + "-certificate.pem": output / slot / "identity/certificate.pem" for slot in network.SLOTS},
        "bootstrap-certificate.pem": output / "seed/bootstrap/certificate.pem",
    }.items():
        private_write(public_directory / name, source.read_bytes())
    receipt = {
        "schema": "organism-lineage-owner-enrollment-v1", "release_id": catalog["release_id"],
        "policy_id": policy.policy_id, "book_id": book.book_id, "device_ids": ids,
        "owner_public": base64.b64encode(owner_public).decode(),
        "execution_slots": list(network.SLOTS), "owner_transport_principal": ids["bootstrap"],
        "bootstrap_has_execution_slot": False, "bundles": bundles,
        "public_files": {path.name: file_identity(path) for path in sorted(public_directory.iterdir())},
        "owner_private_key_serialized": False, "receiver_started": False,
        "descendant_started": False, "native_execution_verified": False, "qualification_passed": False,
    }
    private_write(output / "receipt.json", canonical(receipt))
    return receipt


def verify(directory, slot, catalog):
    """Read an exactly bound private node bundle before owner provisioning."""
    network.slot_value(slot)
    validate_catalog(catalog)
    directory = Path(directory)
    root = directory.lstat()
    require(stat.S_ISDIR(root.st_mode) and not root.st_mode & 0o077,
            "Node bundle root is not a private directory")
    names = {"node.json", "enrollment.json", "discovery.json"} | {
        "identity/" + name for name in ("key.pem", "certificate.pem", "owner.pem")}
    if slot == "seed":
        names |= {"bootstrap/" + name for name in ("key.pem", "certificate.pem", "owner.pem")}
    seen = set()
    for path in directory.rglob("*"):
        entry = path.lstat()
        require(stat.S_ISDIR(entry.st_mode) or stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1,
                "Node bundle contains a link or special entry")
        require(not entry.st_mode & 0o077, "Private node bundle permissions are too broad")
        if path.is_file():
            require(entry.st_size <= MAX_BUNDLE, "Node bundle file exceeds its bound")
            seen.add(str(path.relative_to(directory)))
    require(seen == names, "Node identity bundle has missing or unexpected payloads")
    node = json.loads((directory / "node.json").read_bytes())
    require(set(node) == {
        "schema", "slot", "device_id", "release_id", "policy_id", "owner_public", "device_ids",
        "execution_slots", "bootstrap_has_execution_slot", "underlay_address", "bind_address",
        "initial_owner_mode", "qualification_passed",
    } and node["schema"] == "organism-lineage-owner-node-v1"
        and node["slot"] == slot and node["release_id"] == catalog["release_id"]
        and node["execution_slots"] == list(network.SLOTS)
        and node["bootstrap_has_execution_slot"] is False
        and node["underlay_address"] == network.UNDERLAY[slot]
        and node["bind_address"] == network.ADDRESSES[slot]
        and node["initial_owner_mode"] == "stopped" and node["qualification_passed"] is False,
        "Node bundle changes the admitted slot, release or initial owner fence")
    ids = node["device_ids"]
    require(set(ids) == {*network.SLOTS, "bootstrap"} and len(set(ids.values())) == 4
            and node["device_id"] == ids[slot], "Node identity roles differ")
    public = base64.b64decode(node["owner_public"], validate=True)
    policy = Enrollment((directory / "enrollment.json").read_bytes(), public)
    require(policy.policy_id == node["policy_id"] and set(policy.body["devices"]) == set(ids.values())
            and all(policy.platform(identity) == "linux-x86_64" for identity in ids.values()),
            "Node enrollment differs from the three slots and owner principal")
    manifest = json.loads(base64.b64decode(catalog["manifest_base64"], validate=True))
    for name, expected in {
        "experiment_id": manifest["body"]["experiment_id"], "epoch": 1,
        "release_keys": catalog["release_keys"], "require_trained": True,
        "max_payload_bytes": 3 * 1024**3, "birth_seconds": 1200,
        "max_births_per_device": 1, "max_process_starts": 1,
    }.items():
        require(type(policy.body[name]) is type(expected) and policy.body[name] == expected,
                "Node enrollment changed " + name)
    book = DiscoveryBook((directory / "discovery.json").read_bytes(), policy, public)
    require(book.body["lab_cidrs"] == [network.LAB] and book.body["epoch"] == 1
            and book.body["endpoints"] == {
                ids[name]: {"address": value, "port": 5443}
                for name, value in {**network.ADDRESSES, "bootstrap": network.BOOTSTRAP_ADDRESS}.items()},
            "Node discovery endpoints differ from the fixed laboratory")
    context(policy, ids[slot], directory / "identity")
    if slot == "seed":
        context(policy, ids["bootstrap"], directory / "bootstrap")
    return node, policy, book
