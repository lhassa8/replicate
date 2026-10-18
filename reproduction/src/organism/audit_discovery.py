"""Independent readback of a public Mac discovery probe, not qualification."""

import argparse
import base64
from datetime import datetime
import ipaddress
import json
from pathlib import Path
import sqlite3

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .audit_application import audit as audit_application, canonical, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def moment(value):
    result = datetime.fromisoformat(value)
    require(result.tzinfo is not None, "Evidence timestamp has no timezone")
    return result


def signed_body(envelope, public):
    require(set(envelope) == {"body", "signature", "owner_id"}, "Unexpected signed envelope")
    require(envelope["owner_id"] == sha(public), "Owner identity differs")
    Ed25519PublicKey.from_public_bytes(public).verify(
        base64.b64decode(envelope["signature"], validate=True), canonical(envelope["body"]))
    return envelope["body"]


def chain(records):
    previous = "0" * 64
    require(0 < len(records) <= 65536, "Missing or oversized attempt history")
    for sequence, item in enumerate(records, 1):
        data = canonical(item["record"])
        require(len(data) <= 2048 and item["sequence"] == sequence, "Audit sequence or size differs")
        require(item["digest"] == sha(bytes.fromhex(previous) + data), "Audit chain differs")
        previous = item["digest"]
    return previous


def anchored(records, receipt):
    sequence = receipt["sequence"]
    require(type(sequence) is int and 1 <= sequence <= len(records), "Receipt sequence is outside history")
    item = records[sequence - 1]
    require(item["digest"] == receipt["digest"], "Receipt hash differs")
    return item["record"]


def read_database(path, exported):
    """Read the closed copy without importing or opening the runtime writer."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        rows = db.execute("SELECT sequence,previous,digest,body FROM records ORDER BY sequence").fetchall()
        head = db.execute("SELECT sequence,digest FROM head WHERE singleton=1").fetchone()
    actual, previous = [], "0" * 64
    for sequence, ancestor, digest, body in rows:
        require(ancestor == previous, "Database predecessor differs")
        record = json.loads(body)
        require(body.encode() == canonical(record), "Database record encoding differs")
        actual.append({"sequence": sequence, "digest": digest, "record": record})
        previous = digest
    require(actual == exported, "Export differs from retained database")
    require(head == (len(actual), chain(actual)), "Database head differs")


def certificate(raw, identity, device, public):
    cert = x509.load_pem_x509_certificate(raw.encode())
    key = cert.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    require(sha(key) == identity and key == base64.b64decode(device["public_key"], validate=True),
            "Certificate key differs from enrolled device")
    Ed25519PublicKey.from_public_bytes(public).verify(cert.signature, cert.tbs_certificate_bytes)
    require(not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca,
            "Peer certificate is a CA")
    names = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    require(names.get_values_for_type(x509.UniformResourceIdentifier) == ["urn:organism:device:" + identity]
            and names.get_values_for_type(x509.DNSName) == [f"d-{identity[:32]}.{identity[32:]}.organism.invalid"],
            "Certificate names differ from device identity")
    return cert, sha(cert.public_bytes(serialization.Encoding.DER))


def inspect(directory):
    directory = Path(directory).resolve()
    run = json.loads((directory / "report.json").read_bytes())
    report = json.loads((directory / "discovery.json").read_bytes())
    require(run["with_discovery"] and run["sources_unchanged_during_run"], "Discovery run or source continuity missing")
    require(not any(run[name] for name in ("trained_for_organism", "native_execution_verified",
                                         "autonomous_reproduction_verified", "qualification_passed")),
            "Development probe incorrectly claims qualification")
    require(not any(report[name] for name in ("native_execution_verified", "autonomous_reproduction_verified",
                                             "private_keys_exported")), "Discovery scope differs")
    require(report["poll_seconds"] == 60 and report["probe_seconds"] == 5, "Default discovery limits differ")
    for name, expected in run["source_hashes"].items():
        require(Path(name).name == name and name.endswith(".py"), "Invalid source snapshot name")
        require(sha((directory / "source-snapshot" / name).read_bytes()) == expected, "Source snapshot differs")
    application = audit_application(directory)
    public = base64.b64decode(report["public_owner_key"], validate=True)
    policy = signed_body(report["signed_enrollment"], public)
    book = signed_body(report["signed_discovery_book"], public)
    policy_id, book_id = sha(canonical(report["signed_enrollment"])), sha(canonical(report["signed_discovery_book"]))
    local, existing, new = (report[name] for name in ("local_device_id", "existing_peer_id", "new_peer_id"))
    require(len({local, existing, new}) == 3 and set(policy["devices"]) == {local, existing, new}
            and set(book["endpoints"]) == set(policy["devices"]), "Probe roster differs")
    require(policy["schema"] == "organism-enrollment-v1" and book["schema"] == "organism-discovery-book-v1"
            and book["policy_id"] == policy_id, "Signed schema or policy binding differs")
    for identity, device in policy["devices"].items():
        require(sha(base64.b64decode(device["public_key"], validate=True)) == identity, "Enrolled device key differs")
    require(book["lab_cidrs"] == ["127.0.0.0/24"], "Mac probe escaped its declared loopback network")
    endpoints = book["endpoints"]
    for endpoint in endpoints.values():
        require(ipaddress.ip_address(endpoint["address"]) in ipaddress.ip_network("127.0.0.0/24")
                and type(endpoint["port"]) is int and 1 <= endpoint["port"] <= 65535, "Invalid loopback endpoint")
    require(len({(item["address"], item["port"]) for item in endpoints.values()}) == 3, "Duplicate endpoint")
    marker = json.loads((directory / "work-evidence/discovery/discovery.json").read_bytes())
    require(marker == {"schema": "organism-discovery-state-v1", "book_id": book_id,
                       "policy_id": policy_id, "local_device_id": local}, "Discovery state identity differs")
    health = [run["initial_health"], report["application_health_before_arrival"],
              report["application_health_after_arrival"], run["final_health"]]
    token = health[0]["start_token"]
    require(all(item["status"] == "serving" and item["inference_ready"] and item["discovery_running"]
                and item["start_token"] == token and not item["native"]
                and not item["useful_readiness_verified"] for item in health), "Application continuity differs")
    receipt = json.loads((directory / "work-evidence" / ("application-" + token + ".json")).read_bytes())
    require(receipt["device_id"] == local and receipt["start_token"] == token
            and receipt["shutdown"] == run["shutdown"], "Application identity or shutdown differs")
    records = report["observation_attempts"]
    chain(records)
    read_database(directory / "work-evidence/discovery/peer-audit.sqlite3", records)
    require(set(report["receivers"]) == {existing, new}, "Receiver evidence is missing")
    peer_records, certificates = {}, {}
    for identity, receiver in report["receivers"].items():
        require(receiver["current_birth"] is None and receiver["events"] == [], "Read-only probe changed a birth")
        peer_records[identity] = receiver["transport_records"]
        chain(peer_records[identity])
        certificates[identity] = certificate(receiver["certificate_pem"], identity, policy["devices"][identity], public)
    started, terminal, linked = {}, set(), set()
    times = {existing: [], new: []}
    lower, arrival, upper = moment(run["started_at"]), moment(report["arrival_at"]), moment(run["completed_at"])
    require(lower < arrival < upper, "Arrival falls outside application run")
    for item in records:
        record = item["record"]
        peer, detail, request = record["peer_id"], record["detail"], record["request_id"]
        require(peer in peer_records and record["operation"] == "discover" and record["operation_id"] is None
                and detail["start_token"] == token and detail["boot_id"] == "mac-development-observer",
                "Discovery attempt identity or operation differs")
        when = moment(record["recorded_at"])
        require(lower <= when <= upper, "Discovery attempt falls outside run")
        if record["phase"] == "started":
            require(request not in started and detail["endpoint"] == [
                endpoints[peer]["address"], endpoints[peer]["port"]], "Attempt start or destination differs")
            started[request] = record
            times[peer].append(when)
            continue
        require(request in started and request not in terminal and started[request]["peer_id"] == peer
                and moment(started[request]["recorded_at"]) <= when, "Attempt completion lacks matching start")
        terminal.add(request)
        if record["phase"] != "observed":
            require(record["phase"] in ("unreachable", "interrupted") and detail["observation"] is None,
                    "Unexpected discovery failure")
            continue
        observation = detail["observation"]
        cert, cert_hash = certificates[peer]
        require(detail["authenticated_device_id"] == peer and detail["tls"] == "TLSv1.3"
                and detail["certificate_sha256"] == cert_hash
                and cert.not_valid_before_utc <= when <= cert.not_valid_after_utc, "Authenticated certificate differs")
        require(observation["device_id"] == peer and observation["policy_id"] == policy_id
                and observation["platform"] == policy["devices"][peer]["platform"]
                and observation["revision"] == 0 and observation["owner_mode"] == "enabled"
                and observation["available"] is True and observation["observation_valid_seconds"] == 30
                and "admission_ticket" not in observation, "Read-only admission observation differs")
        anchor = observation["transport_receipt"]
        remote = anchored(peer_records[peer], anchor)
        preceding = [row["record"] for row in peer_records[peer][:anchor["sequence"] - 1]
                     if row["record"]["request_id"] == remote["request_id"] and row["record"]["phase"] == "started"]
        require(remote["phase"] == "completed" and remote["operation"] == "observe" and remote["peer_id"] == local
                and remote["operation_id"] is None and len(preceding) == 1
                and preceding[0]["peer_id"] == local and preceding[0]["operation"] == "observe"
                and preceding[0]["detail"]["tls"] == "TLSv1.3"
                and moment(started[request]["recorded_at"]) <= moment(remote["recorded_at"]) <= when,
                "Receiver completion does not match observation")
        require((peer, anchor["sequence"]) not in linked, "Receiver completion reused")
        linked.add((peer, anchor["sequence"]))
    require(terminal == set(started), "Unfinished discovery attempts")
    for peer, checks in times.items():
        require(len(checks) >= 2 and all((b - a).total_seconds() >= 59 for a, b in zip(checks, checks[1:])),
                "Discovery history does not show the normal one-minute interval")
        require(all(row["record"]["operation"] == "observe" and row["record"]["operation_id"] is None
                    and row["record"]["peer_id"] == local and row["record"]["phase"] in ("started", "completed")
                    for row in peer_records[peer]), "Unexpected receiver operation")
        require(sum(row["record"]["phase"] == "completed" for row in peer_records[peer])
                == sum(identity == peer for identity, _ in linked), "Unmatched receiver completions")
    snapshots = {}
    for label in ("before_arrival", "after_arrival"):
        rows = report[label]
        require(len(rows) == 2 and {row["device_id"] for row in rows} == {existing, new}, "Snapshot roster differs")
        snapshots[label] = {}
        for row in rows:
            record = anchored(records, row["audit_receipt"])
            require(record["peer_id"] == row["device_id"] and record["detail"]["observation"] == row["observation"]
                    and row["admission_requires_fresh_observation"] is True and 0 <= row["age_seconds"] < 30,
                    "Snapshot differs from its current observation")
            expected_state = "current" if row["observation"] is not None else "unreachable"
            require(row["state"] == expected_state and row["available_observed"] == (expected_state == "current"),
                    "Snapshot availability differs")
            snapshots[label][row["device_id"]] = record
    before, after = snapshots["before_arrival"], snapshots["after_arrival"]
    require(before[new]["phase"] == "unreachable" and before[existing]["phase"] == "observed"
            and moment(before[new]["recorded_at"]) < arrival < moment(after[new]["recorded_at"])
            and after[new]["phase"] == "observed" and after[existing]["phase"] == "observed",
            "New arrival was not discovered after a recorded unavailable state")
    observed_delay = (moment(after[new]["detail"]["observed_at"]) - arrival).total_seconds()
    require(0 < report["discovered_after_seconds"] <= 80
            and abs(report["discovered_after_seconds"] - observed_delay) < 1, "Discovery delay differs")
    require(all(b"-----BEGIN PRIVATE KEY-----" not in path.read_bytes()
                for path in (directory / "work-evidence").rglob("*") if path.is_file()),
            "Retained runtime evidence contains a PEM private key")
    return {"scope": "Independent Mac artifact readback; public receiver fixtures, not native or lineage qualification",
            "trace_consistency_passed": True, "same_application_incarnation": True,
            "authenticated_observations_cross_referenced": len(linked), "recorded_attempts": len(started),
            "new_enrolled_receiver_discovered": True, "discovered_after_seconds": report["discovered_after_seconds"],
            "application_task_successes": application["task_successes"], "application_task_trials": application["task_trials"],
            "source_snapshots_verified": len(run["source_hashes"]), "births_created": 0,
            "runtime_evidence_contains_pem_private_key": False,
            "limits": ["Local exported evidence is not immutable against a privileged host",
                       "Observer certificate was not separately exported; its key binding relies on the recorded TLS authentication",
                       "Native execution, complete reproduction and whole-device resource use remain unverified"],
            "qualification_passed": False, "auditor_sha256": sha(Path(__file__).read_bytes()),
            "application_auditor_sha256": application["auditor_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = inspect(args.directory)
    with (args.directory / "discovery-audit.json").open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
