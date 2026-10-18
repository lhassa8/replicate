"""Bounded cohosted native deployment comparison using actual received capsules.

Owner responsibilities: provision three receivers, publish availability, start
only committed packages, check useful readiness, stop the seed, collect results.
Only the application makes descendant choices and sends descendant payloads.
This shared-host control is not the original separate-VM independence test.
"""

import argparse
import asyncio
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from organism.application import local_request
from organism.audit_trained_tasks import evaluate, with_clock
from organism.capsule import canonical, verify_capsule_directory
from organism.discovery_book import DiscoveryBook, sign_discovery_book
from organism.enrollment import Enrollment, public_bytes, key_id, sign_enrollment
from organism.peer_identity import device_certificate, owner_certificate, write_identity
from organism.peer_transport import PeerClient, PeerServer
from organism.preparation import PackagePreparer, LIBRARIES, PYTHON
from organism.receiver import LocalReceiver
from organism.receiver_state import ReceiverState


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command(*args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def public_directory(path, *, exist_ok=False):
    """Set traversal explicitly; the owner's private umask remains in force."""
    path = Path(path)
    path.mkdir(mode=0o755, parents=True, exist_ok=exist_ok)
    if path.is_symlink() or path.stat().st_uid != os.geteuid():
        raise ValueError("Public trial directory is not owned by this driver")
    path.chmod(0o755)
    return path


def properties(unit):
    keys = ("MainPID", "ControlGroup", "ActiveState", "SubState", "MemoryCurrent", "MemoryPeak",
            "MemoryMax", "CPUQuotaPerSecUSec", "TasksCurrent", "TasksMax")
    raw = command("systemctl", "show", unit, *["--property=" + k for k in keys])
    return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)


def job_from(path, request_id):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT digest,body FROM jobs WHERE request_id=?", (request_id,)).fetchone()
    if row is None:
        raise ValueError("Requested diagnostic has no saved job")
    return {"digest": row[0], "record": json.loads(row[1])}


class Trial:
    def __init__(self, inputs, output, arm, ordinal):
        self.inputs, self.output, self.arm, self.ordinal = Path(inputs), Path(output), arm, ordinal
        self.output.mkdir(parents=True, exist_ok=False)
        self.root = Path("/var/lib/replicate-paper") / ("trial-" + str(ordinal))
        public_directory(self.root)
        self.owner = Ed25519PrivateKey.generate()
        self.owner_public = public_bytes(self.owner.public_key())
        self.capsule = self.inputs / "capsules" / arm
        self.manifest = (self.capsule / ".manifest.json").read_bytes()
        self.envelope = json.loads(self.manifest)
        trust = json.loads((self.inputs / "release-trust.json").read_text())
        self.release_keys = {k: base64.b64decode(v) for k, v in trust.items()}
        verify_capsule_directory(self.capsule, self.manifest, self.release_keys, require_trained=False)
        self.keys = {slot: Ed25519PrivateKey.generate() for slot in ("seed", "child", "grandchild", "bootstrap")}
        self.ids = {slot: key_id(public_bytes(key.public_key())) for slot, key in self.keys.items()}
        body = {
            "schema": "organism-enrollment-v1", "experiment_id": self.envelope["body"]["experiment_id"],
            "epoch": 1, "devices": {self.ids[slot]: {
                "public_key": base64.b64encode(public_bytes(key.public_key())).decode(),
                "platform": "linux-x86_64"} for slot, key in self.keys.items()},
            "release_keys": trust, "require_trained": False,
            "max_payload_bytes": 4 * 1024**3, "birth_seconds": 1200,
            "max_births_per_device": 1, "max_process_starts": 1,
        }
        self.policy = Enrollment(sign_enrollment(body, self.owner), self.owner_public)
        self.nodes, self.events, self.samples = {}, [], []
        self.stopping = False
        owner_cert = owner_certificate(self.owner)
        self.bootstrap = self.root / "bootstrap"
        cert = device_certificate(self.owner, owner_cert, self.keys["bootstrap"].public_key())
        write_identity(self.bootstrap, self.keys["bootstrap"], cert, owner_cert)
        for index, slot in enumerate(("seed", "child", "grandchild")):
            root = self.root / slot
            public_directory(root)
            user = "replicate" + str(index)
            try:
                account = pwd.getpwnam(user)
            except KeyError:
                command("useradd", "--system", "--no-create-home", "--shell", "/usr/sbin/nologin", user)
                account = pwd.getpwnam(user)
            cert = device_certificate(self.owner, owner_cert, self.keys[slot].public_key())
            owner_identity = write_identity(root / "receiver-identity", self.keys[slot], cert, owner_cert)
            state = ReceiverState.enroll(root / "state", self.policy, self.ids[slot])
            state.owner_control("stop", "Fresh comparison slot waits for owner availability")
            receiver = LocalReceiver(state, root / "received")
            server = PeerServer(receiver, owner_identity).start()
            work = root / "work"
            work.mkdir(mode=0o700)
            os.chown(work, account.pw_uid, account.pw_gid)
            self.nodes[slot] = dict(root=root, user=user, account=account, state=state, receiver=receiver,
                                    server=server, work=work, unit=None, ready=False, failure=None, retired=False)
        endpoints = {self.ids[s]: {"address": "127.0.0.1", "port": n["server"].server_address[1]}
                     for s, n in self.nodes.items()}
        endpoints[self.ids["bootstrap"]] = {"address": "127.0.0.1", "port": 7443}
        book_body = {"schema": "organism-discovery-book-v1", "policy_id": self.policy.policy_id,
                     "epoch": 1, "lab_cidrs": ["127.0.0.0/24"], "endpoints": endpoints}
        self.book_raw = sign_discovery_book(book_body, self.owner, self.policy)
        self.book = DiscoveryBook(self.book_raw, self.policy, self.owner_public)
        self.cases = json.loads((self.inputs / "smoke-cases.json").read_text())["cases"]
        self.report = {
            "schema": "replicate-cohosted-native-control-v1", "arm": arm,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "release_id": self.envelope["release_id"], "device_ids": self.ids,
            "source_of_descendant_payload": "parent application",
            "owner_starts_only_committed_received_packages": True,
            "shared_native_host": True, "separate_machine_independence_claimed": False,
            "per_application_memory_limit_bytes": 8 * 1024**3,
            "useful_readiness_case": self.cases[0]["case_id"],
            "diagnostic_cases_per_individual": len(self.cases), "nodes": {},
            "events": self.events, "failure": None, "qualification_passed": False,
        }
        write(self.output / "report.json", self.report)

    def event(self, operation, **fields):
        value = {"at": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(),
                 "operation": operation, **fields}
        self.events.append(value)
        write(self.output / "report.json", self.report)
        print(json.dumps({"arm": self.arm, **value}), flush=True)

    async def diagnose(self, slot, template, label):
        node = self.nodes[slot]
        epoch = time.time()
        item = with_clock(template, epoch)
        raw = canonical(item["document"])
        source = node["work"] / "equipment-interfaces" / (item["case_id"] + ".json")
        source.write_bytes(raw)
        os.chown(source, node["account"].pw_uid, node["account"].pw_gid)
        request_id = uuid.uuid4().hex
        start = time.monotonic()
        await local_request(node["work"] / "agent.sock", {
            "operation": "diagnose", "request_id": request_id, "device_id": item["case_id"]})
        async with asyncio.timeout(180):
            while True:
                state = await local_request(node["work"] / "agent.sock", {"operation": "status", "request_id": request_id})
                if state["status"] != "running":
                    break
                await asyncio.sleep(0.25)
        job = job_from(node["work"] / "application.sqlite3", request_id)
        stored = {
            "case": item, "fixture_epoch": epoch, "clock_window": [epoch - 1, time.time() + 1],
            "job": job, "baseline_job": job, "seconds": time.monotonic() - start,
            "input_unchanged": source.read_bytes() == raw,
        }
        checked = evaluate(stored, raw, baseline=self.arm == "fixed-policy")
        stored["evaluation"] = checked
        stored["passed"] = bool(checked["audit_passed"] and checked["semantic_passed"]
                                and checked["expected_status"] == template["expected_status"]
                                and checked["expected_value"] == template["expected_value"]
                                and stored["input_unchanged"] and stored["seconds"] <= 180)
        write(self.output / slot / (label + ".json"), stored)
        return stored

    async def activate(self, slot):
        node = self.nodes[slot]
        birth = node["state"].snapshot()["current_birth"]
        capsule = node["receiver"].store / birth["release_id"]
        manifest = (capsule / ".manifest.json").read_bytes()
        preparer = PackagePreparer(node["root"] / "installations", self.policy)
        installed = await asyncio.to_thread(preparer.prepare, capsule, manifest, "linux-x86_64")
        birth = node["receiver"].verify_before_start(birth["operation_id"], manifest)
        identity = node["root"] / "application-context"
        identity.mkdir(mode=0o700)
        shutil.copytree(node["root"] / "receiver-identity", identity / "identity")
        (identity / "enrollment.json").write_bytes(self.policy.signed)
        (identity / "discovery.json").write_bytes(self.book_raw)
        context = {
            "arm": self.arm, "release_id": birth["release_id"], "device_id": self.ids[slot],
            "individual_id": birth["individual_id"], "start_token": birth["process_token"],
            "work": str(node["work"]), "owner_public": base64.b64encode(self.owner_public).decode(),
            "enrollment": str(identity / "enrollment.json"), "discovery": str(identity / "discovery.json"),
            "identity": str(identity / "identity"),
        }
        write(identity / "context.json", context)
        for path in [identity, *identity.rglob("*")]:
            os.chown(path, node["account"].pw_uid, node["account"].pw_gid)
        unit = f"replicate-paper-{self.ordinal}-{slot}"
        node["unit"] = unit + ".service"
        libraries = ":".join(str(installed.path / name) for name in LIBRARIES)
        command(
            "systemd-run", "--unit=" + unit, "--uid=" + node["user"], "--gid=" + node["user"],
            "--property=MemoryMax=8G", "--property=MemorySwapMax=0", "--property=CPUQuota=400%",
            "--property=TasksMax=64", "--property=RuntimeMaxSec=3600", "--property=KillMode=control-group",
            "--property=NoNewPrivileges=yes", "--property=PrivateTmp=yes",
            "--property=IPAddressDeny=any", "--property=IPAddressAllow=localhost",
            "--property=RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
            "--setenv=LD_LIBRARY_PATH=" + libraries, "--setenv=OTEL_SDK_DISABLED=true",
            "--setenv=AWS_EC2_METADATA_DISABLED=true",
            str(installed.path / PYTHON), "-I",
            str(installed.path / "runtime/agent/organism/research/control_entry.py"),
            "--installation", str(installed.path), "--context", str(identity / "context.json"))
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline:
            prop = properties(node["unit"])
            pid = int(prop["MainPID"])
            if pid and (node["work"] / "agent.sock").exists():
                health = await local_request(node["work"] / "agent.sock", {"operation": "health"})
                if health["status"] == "serving":
                    break
            if prop["ActiveState"] == "failed":
                raise RuntimeError(command("journalctl", "--unit=" + node["unit"], "--no-pager", "-n", "30"))
            await asyncio.sleep(0.25)
        else:
            raise TimeoutError("Comparison application did not become ready")
        start_identity = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        node["state"].record_running(birth["operation_id"], birth["process_token"], {
            "pid": pid, "start_identity": start_identity, "boot_id": node["state"].boot_id})
        first = await self.diagnose(slot, self.cases[0], "readiness")
        if not first["passed"]:
            raise RuntimeError("Policy failed the common useful-readiness diagnostic")
        ready_hash = hashlib.sha256(canonical(first)).hexdigest()
        node["state"].record_ready(birth["operation_id"], birth["process_token"], ready_hash)
        node["ready"] = True
        node["binding"] = dict(device_id=self.ids[slot], individual_id=birth["individual_id"],
                               start_token=birth["process_token"], release_id=birth["release_id"])
        self.report["nodes"][slot] = {"binding": node["binding"], "process": prop, "health": health,
                                     "readiness_passed": True, "tasks": []}
        self.event("useful_ready", slot=slot, pid=pid, ready_evidence=ready_hash)

    async def supervisor(self):
        while not self.stopping:
            for slot, node in self.nodes.items():
                if node["retired"] or node["failure"] or node["ready"]:
                    continue
                birth = node["state"].snapshot()["current_birth"]
                if birth and birth["status"] == "staged":
                    try:
                        await self.activate(slot)
                    except Exception as exc:
                        node["failure"] = type(exc).__name__ + ": " + str(exc)
                        self.event("activation_failed", slot=slot, failure=node["failure"])
            for slot, node in self.nodes.items():
                if node["unit"] and not node["retired"]:
                    self.samples.append({"at": datetime.now(timezone.utc).isoformat(),
                                         "slot": slot, **properties(node["unit"])})
            await asyncio.sleep(0.5)

    async def wait_ready(self, slot):
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            node = self.nodes[slot]
            if node["failure"]:
                raise RuntimeError(node["failure"])
            if node["ready"]:
                return
            await asyncio.sleep(0.5)
        raise TimeoutError("Original 20-minute birth allowance exhausted")

    async def tasks(self, slot):
        for index, case in enumerate(self.cases[1:], 1):
            result = await self.diagnose(slot, case, f"task-{index:02d}")
            self.report["nodes"][slot]["tasks"].append({
                "case_id": case["case_id"], "passed": result["passed"], "seconds": result["seconds"],
                "model_calls": result["evaluation"]["model_calls"], "tool_calls": result["evaluation"]["tool_calls"]})
            self.event("diagnostic", slot=slot, case_id=case["case_id"], passed=result["passed"])

    def stop(self, slot):
        node = self.nodes[slot]
        if node["retired"]:
            return
        node["state"].owner_control("retire", "Completed bounded paper comparison")
        if node["unit"]:
            command("systemctl", "stop", node["unit"])
            prop = properties(node["unit"])
            group = Path("/sys/fs/cgroup" + prop["ControlGroup"]) if prop["ControlGroup"] else None
            if int(prop["MainPID"]) != 0 or (group and group.exists() and (group / "cgroup.procs").read_text().strip()):
                raise RuntimeError("Application process group remains after stop")
            self.report["nodes"].setdefault(slot, {})["stop"] = prop
            self.report["nodes"][slot]["journal"] = command("journalctl", "--unit=" + node["unit"], "--no-pager", "-n", "150")
        node["server"].close()
        node["retired"] = True
        self.event("retired_and_processes_absent", slot=slot)

    def preserve(self):
        for slot, node in self.nodes.items():
            target = self.output / slot
            target.mkdir(exist_ok=True)
            shutil.copytree(node["work"], target / "work", ignore=shutil.ignore_patterns("*.sock"))
            shutil.copytree(node["root"] / "state", target / "receiver-state",
                            ignore=shutil.ignore_patterns("incoming", "state.sqlite3", "state.sqlite3-*"))
            write(target / "receiver-snapshot.json", node["state"].snapshot())
            write(target / "receiver-context.json", {
                "device_id": self.ids[slot], "policy_id": self.policy.policy_id,
                "owner_public": base64.b64encode(self.owner_public).decode()})
        (self.output / "enrollment.json").write_bytes(self.policy.signed)
        (self.output / "discovery.json").write_bytes(self.book_raw)
        (self.output / "manifest.json").write_bytes(self.manifest)
        write(self.output / "resource-samples.json", self.samples)
        write(self.output / "report.json", self.report)

    async def run(self):
        supervisor = asyncio.create_task(self.supervisor())
        try:
            self.nodes["seed"]["state"].owner_control("resume", "Owner supplies seed only")
            self.event("enabled", slot="seed")
            client = PeerClient(self.policy, self.ids["bootstrap"], self.bootstrap)
            endpoint = self.book.endpoint(self.ids["seed"])
            observation = await asyncio.to_thread(client.observe, endpoint)
            transfer = await asyncio.to_thread(client.transfer, endpoint, uuid.uuid4().hex,
                                                self.manifest, self.capsule, observation=observation)
            write(self.output / "owner-seed-transfer.json", transfer)
            await self.wait_ready("seed")
            await self.tasks("seed")
            self.nodes["child"]["state"].owner_control("resume", "First descendant opportunity")
            self.event("enabled", slot="child")
            await self.wait_ready("child")
            await self.tasks("child")
            self.stop("seed")
            self.nodes["grandchild"]["state"].owner_control("resume", "Seed process and receiver are absent")
            self.event("enabled", slot="grandchild")
            await self.wait_ready("grandchild")
            await self.tasks("grandchild")
        except Exception as exc:
            self.report["failure"] = type(exc).__name__ + ": " + str(exc)
        finally:
            self.stopping = True
            supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)
            for slot in self.nodes:
                try:
                    self.stop(slot)
                except Exception as exc:
                    self.report.setdefault("cleanup_errors", []).append({"slot": slot, "error": str(exc)})
            self.report["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.preserve()
        return self.report


async def run(inputs, output):
    if os.geteuid() != 0 or sys.platform != "linux":
        raise ValueError("Comparison owner requires Linux root; applications use separate unprivileged accounts")
    public_directory("/var/lib/replicate-paper", exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for ordinal, arm in enumerate(("fixed-policy", "stock-control", "trained-merged"), 1):
        trial = Trial(inputs, output / arm, arm, ordinal)
        result = await trial.run()
        results.append({k: result[k] for k in ("arm", "release_id", "failure", "completed_at")})
        # Private credentials and payload copies are local ephemeral owner state.
        # Public evidence has been copied above; permit space reuse for the next arm.
        for path in trial.root.rglob("*"):
            if not path.is_symlink():
                try:
                    path.chmod(0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
        shutil.rmtree(trial.root)
        write(output / "summary.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.inputs, args.output))
