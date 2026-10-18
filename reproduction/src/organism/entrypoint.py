"""Native installed entry point, invoked by portable Python with -I.

Bootstrap uses only the standard library until the immutable owner-installed
tree and current systemd incarnation have been checked. The receiving substrate
is responsible for the preceding owner-signature and process-start admission.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_installation(root, *, owner_uid=0):
    root = Path(root).absolute()
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_uid or stat.S_IMODE(info.st_mode) != 0o555:
        raise ValueError("Installed root must be immutable and owned by the receiving service")
    index_path = root / ".installation.json"
    info = index_path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != owner_uid or info.st_nlink != 1
            or info.st_size > 16 * 1024**2 or stat.S_IMODE(info.st_mode) != 0o444):
        raise ValueError("Invalid owner-installed runtime index")
    index = json.loads(index_path.read_bytes())
    inventory = index.get("inventory")
    if index.get("schema") != "organism-prepared-installation-v1" or not isinstance(inventory, dict) or len(inventory) > 60000:
        raise ValueError("Invalid installed runtime schema")
    observed = set()
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(parent) / name
            relative = path.relative_to(root).as_posix()
            if relative == ".installation.json":
                continue
            if relative not in inventory:
                raise ValueError("Unlisted installed runtime entry")
            observed.add(relative)
            expected, info = inventory[relative], path.lstat()
            if info.st_uid != owner_uid:
                raise ValueError("Installed runtime entry has another owner")
            mode = stat.S_IMODE(info.st_mode)
            if expected["type"] == "symlink":
                if (not stat.S_ISLNK(info.st_mode) or os.readlink(path) != expected["target"]
                        or not path.resolve(strict=True).is_relative_to(root.resolve())):
                    raise ValueError("Installed runtime link differs")
            elif expected["type"] == "directory":
                if not stat.S_ISDIR(info.st_mode) or mode != 0o555:
                    raise ValueError("Installed runtime directory differs")
            elif expected["type"] == "file":
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or mode != expected["mode"]
                        or mode not in (0o444, 0o555) or info.st_size != expected["bytes"] or digest(path) != expected["sha256"]):
                    raise ValueError("Installed runtime file differs")
            else:
                raise ValueError("Unexpected installed runtime entry type")
    if observed != set(inventory):
        raise ValueError("Installed runtime entries are missing")
    manifest_path = root / "capsule/.manifest.json"
    if digest(manifest_path) != index["manifest_sha256"]:
        raise ValueError("Installed capsule manifest differs")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["release_id"] != index["release_id"]:
        raise ValueError("Installed release identity differs")
    model = manifest["body"]["model"]
    if type(model["parameters"]) is not int or not 2_000_000_000 <= model["parameters"] <= 4_000_000_000:
        raise ValueError("Installed model declaration exceeds the required size class")
    return index, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--installation", required=True, type=Path)
    parser.add_argument("--work-directory", required=True, type=Path)
    parser.add_argument("--node-context", type=Path)
    parser.add_argument("--node-context-sha256")
    args = parser.parse_args()
    if platform.system() != "Linux" or os.geteuid() == 0:
        raise RuntimeError("Native organism entry requires an unprivileged Linux service")
    if args.node_context is None or not args.node_context_sha256:
        raise ValueError("Native individual requires its own admitted discovery context")
    root = args.installation.resolve()
    if Path(__file__).resolve() != root / "runtime/agent/organism/entrypoint.py":
        raise ValueError("Entry point is outside its installed capsule")
    token, device, individual = (os.environ.get(name, "") for name in
                                  ("ORGANISM_START_TOKEN", "ORGANISM_DEVICE_ID", "ORGANISM_INDIVIDUAL_ID"))
    if (not re.fullmatch("[0-9a-f]{32}", token) or not re.fullmatch("[0-9a-f]{64}", device)
            or not re.fullmatch("[0-9a-f]{32}", individual)):
        raise ValueError("Owner-supplied application identity is missing")
    unit = "organism-" + device[:16] + "-" + individual + "-" + token + ".service"
    # The owner renderer selects the service's exact parent. Keep this check
    # stdlib-only until the installed code has passed its inventory check.
    expected_group = os.environ.get("ORGANISM_SERVICE_CGROUP", "")
    allowed_groups = tuple(parent + "/" + unit for parent in
                           ("/system.slice", "/system.slice/system-organism.slice"))
    if (expected_group not in allowed_groups
            or Path("/proc/self/cgroup").read_text().splitlines() != ["0::" + expected_group]):
        raise ValueError("Application is not in its owner-admitted service group")
    index, manifest = validate_installation(root)
    if index["platform"] != {"x86_64": "linux-x86_64", "aarch64": "linux-arm64"}.get(platform.machine()):
        raise ValueError("Installed runtime differs from the actual native machine")
    work = args.work_directory.absolute()
    if work.resolve().is_relative_to(root):
        raise ValueError("Mutable work must be outside the installation")
    # Admit only the checked application and dependency trees to import search.
    sys.path.insert(0, str(root / "runtime/dependencies/site-packages"))
    sys.path.insert(0, str(root / "runtime/agent"))
    from organism.application import AgentApplication
    from organism.inference_server import RuntimeConfig
    from organism.node_context import load_context
    from organism.discovery import Discovery
    from organism.async_peer import AsyncPeerClient
    from organism.peer_identity import LabNetwork
    from organism.reproduction import Reproduction
    node = load_context(args.node_context, args.node_context_sha256, device_id=device,
                        individual_id=individual, release_id=index["release_id"])
    if node.policy.platform(device) != index["platform"]:
        raise ValueError("Device context platform differs from the installed runtime")
    model = manifest["body"]["model"]
    model_entry = manifest["body"]["files"][model["path"]]
    binary = "runtime/inference/llama-server"
    config = RuntimeConfig(
        root / "capsule" / model["path"], model_entry["sha256"], model_entry["bytes"],
        root / binary, index["inventory"][binary]["sha256"],
        token, device, individual, index["release_id"], model["trained_for_organism"], True,
        tuple(root / name for name in ("runtime/dependencies/supplemental/lib", "runtime/python/lib", "runtime/inference")),
    )

    async def serve():
        discovery = Discovery(work / "discovery", node.book, device, node.tls,
                              boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(), start_token=token)
        client = AsyncPeerClient(node.policy, device, node.path, network=LabNetwork(node.book.body["lab_cidrs"]),
                                 identity_owner_uid=node.owner_uid, identity_read_group=node.group)
        reproduction = Reproduction(config, work, discovery, client, root / "capsule")
        experience = None
        if node.book.experience_checkers:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            from organism.node_context import read_file
            from organism.peer_identity import context
            from organism.experience_store import ExperienceStore
            from organism.experience_exchange import ExperienceExchange
            from organism.native_source_reviewer import installed_reviews
            key = load_pem_private_key(
                read_file(node.path / "key.pem", owner_uid=node.owner_uid, group=node.group), password=None)
            store = ExperienceStore(work / "experience", node.policy, key, release_id=config.release_id,
                                    allowed_checkers=node.book.experience_checkers)
            server_tls = context(node.policy, device, node.path, server=True,
                                 owner_uid=node.owner_uid, read_group=node.group)
            reviews = installed_reviews(root / "runtime/knowledge", work, config, node.policy, key,
                                        node.book.experience_checkers)
            experience = ExperienceExchange(config, store, node.book, node.tls, server_tls, reviews=reviews)
        app = AgentApplication(config, work, discovery=discovery, reproduction=reproduction, experience=experience)
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, app.stop_event.set)
        await app.run()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
