"""Execute a received comparison capsule in its own unprivileged Linux service."""

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import platform
import signal
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64" or os.geteuid() == 0:
        raise ValueError("Control application requires an unprivileged native x86-64 Linux process")
    root = args.installation.resolve()
    if Path(__file__).resolve() != root / "runtime/agent/organism/research/control_entry.py":
        raise ValueError("Comparison entry point must be supplied by the received package")
    sys.path.insert(0, str(root / "runtime/agent"))
    sys.path.insert(0, str(root / "runtime/dependencies/site-packages"))
    from organism.application import AgentApplication
    from organism.async_peer import AsyncPeerClient
    from organism.discovery import Discovery
    from organism.discovery_book import DiscoveryBook
    from organism.enrollment import Enrollment
    from organism.inference_server import RuntimeConfig
    from organism.reproduction import Reproduction
    from organism.research.fixed_lifecycle_candidate import (
        FixedLifecycleCandidate, FixedReproductionLifecycleCandidate,
    )

    context = json.loads(args.context.read_text())
    index = json.loads((root / ".installation.json").read_text())
    manifest = json.loads((root / "capsule/.manifest.json").read_text())
    if context["release_id"] != manifest["release_id"] or index["release_id"] != manifest["release_id"]:
        raise ValueError("Comparison configuration differs from the installed release")
    owner_public = base64.b64decode(context["owner_public"])
    policy = Enrollment(Path(context["enrollment"]).read_bytes(), owner_public)
    book = DiscoveryBook(Path(context["discovery"]).read_bytes(), policy, owner_public)
    if policy.platform(context["device_id"]) != "linux-x86_64":
        raise ValueError("Comparison platform mismatch")
    model = manifest["body"]["model"]
    model_id = manifest["body"]["files"][model["path"]]
    config = RuntimeConfig(
        root / "capsule" / model["path"], model_id["sha256"], model_id["bytes"],
        root / "runtime/inference/llama-server", index["inventory"]["runtime/inference/llama-server"]["sha256"],
        context["start_token"], context["device_id"], context["individual_id"], context["release_id"],
        model["trained_for_organism"], True,
        tuple(root / path for path in ("runtime/dependencies/supplemental/lib", "runtime/python/lib", "runtime/inference")),
    )
    config.verify()
    work = Path(context["work"])

    async def run():
        client = AsyncPeerClient(policy, config.device_id, context["identity"])
        discovery = Discovery(work / "discovery", book, config.device_id, client.tls,
                              boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                              start_token=config.start_token, interval=1)
        fixed = context["arm"] == "fixed-policy"
        reproduction = (FixedReproductionLifecycleCandidate if fixed else Reproduction)(
            config, work, discovery, client, root / "capsule", interval=0.25)
        app = (FixedLifecycleCandidate if fixed else AgentApplication)(
            config, work, discovery=discovery, reproduction=reproduction)
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, app.stop_event.set)
        await app.run()

    asyncio.run(run())


if __name__ == "__main__":
    main()
