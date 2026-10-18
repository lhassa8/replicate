"""Independent readback of the carried worker and its public Mac diagnostic."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from .audit_application import audit as audit_application, canonical, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_runtime_binding(run, challenge, expected):
    require(expected["trained_for_organism"] is True
            and run["trained_for_organism"] is True and run["runtime_binding"] == expected,
            "Trained smoke runtime differs from its admitted plan")
    for name in ("model_path", "model_sha256", "model_bytes", "runtime_path", "runtime_sha256", "release_id"):
        require(challenge[name] == expected[name], "Trained smoke challenge changed " + name)
    require(run["model_sha256"] == expected["model_sha256"]
            and run["runtime_sha256"] == expected["runtime_sha256"],
            "Trained smoke report changed model or runtime")
    require(0 < run["peak_sampled_server_rss_bytes"] <= 6 * 1024**3
            and run["rss_samples"] and all(0 < row["rss_bytes"] <= 6 * 1024**3 for row in run["rss_samples"]),
            "Trained smoke resource evidence is missing or exceeds the local allowance")
    require(run["peak_sampled_server_rss_bytes"] == max(row["rss_bytes"] for row in run["rss_samples"]),
            "Trained smoke peak differs from its resource samples")


def inspect(directory, *, expected_runtime=None):
    directory = Path(directory).resolve()
    run = json.loads((directory / "report.json").read_bytes())
    challenge = json.loads((directory / "challenge.json").read_bytes())
    result = run["smoke_result"]
    token = challenge["process_token"]
    require(len(token) == 32 and all(c in "0123456789abcdef" for c in token), "Invalid challenge incarnation")
    target = directory / "work-evidence/readiness" / token
    require(run["sources_unchanged_during_run"] and not run["failure"] and not run.get("shutdown_failure"),
            "Probe sources changed or execution failed")
    require(all(run[name] is False for name in ("native_execution_verified",
                "autonomous_reproduction_verified", "qualification_passed", "receiver_readiness_acknowledged")),
            "Mac smoke probe claims native or qualification readiness")
    if expected_runtime is None:
        require(run["trained_for_organism"] is False, "A trained smoke probe needs its admitted runtime binding")
    else:
        check_runtime_binding(run, challenge, expected_runtime)
    for name, expected in run["source_hashes"].items():
        require(Path(name).name == name and name.endswith(".py"), "Invalid source snapshot name")
        require(sha((directory / "source-snapshot" / name).read_bytes()) == expected, "Application source snapshot differs")
    source = directory / "source-snapshot/carried-readiness.py"
    require(sha(source.read_bytes()) == run["self_test_component"]["source_sha256"] == result["checker_sha256"],
            "Carried checker differs from the packaged worker")
    require(json.loads((directory / "checker-stdout.json").read_bytes()) == result
            and json.loads((target / "result.json").read_bytes()) == result, "Worker output and saved result differ")
    require((target / "challenge.json").read_bytes() == canonical(challenge)
            and result["challenge_sha256"] == sha(canonical(challenge)), "Worker challenge differs")
    for name in ("challenge_id", "device_id", "individual_id", "process_token", "release_id", "native"):
        require(result[name] == challenge[name], "Worker result belongs to another challenge identity")
    require(result["qualification_passed"] is False and result["receiver_readiness_acknowledged"] is False
            and type(result["passed"]) is bool and run["checker_return_code"] == (0 if result["passed"] else 1),
            "Worker outcome, process exit or admission claims differ")
    expected_files = {"challenge.json", "application-before.json", "observation.json", "reply.json", "job.json", "application-after.json"}
    require(set(result["evidence_files"]) <= expected_files, "Unexpected worker evidence path")
    for name, digest in result["evidence_files"].items():
        require(sha((target / name).read_bytes()) == digest, "Worker evidence hash differs")
    if result["passed"]:
        require(set(result["evidence_files"]) == expected_files and result["failure"] is None,
                "Successful worker is missing evidence")
    observation = json.loads((target / "observation.json").read_bytes())
    expected_value = int.from_bytes(bytes.fromhex(challenge["challenge_id"])[:4], "big") % 61 - 10
    require(observation["device_id"] == "readiness-" + challenge["challenge_id"]
            and observation["value"] == expected_value and observation["unit"] == "C"
            and observation["normal_range_celsius"] == [10, 40]
            and observation["maximum_observation_age_seconds"] == 240, "Public smoke observation differs from its challenge")
    job = json.loads((target / "job.json").read_bytes())
    case = json.loads((directory / (observation["device_id"] + ".json")).read_bytes())
    require(case["job"] == job and case["observation"] == observation
            and case["request_id"] == challenge["challenge_id"], "Probe diagnostic differs from worker evidence")
    created = datetime.fromisoformat(job["record"]["created_at"]).timestamp()
    started = datetime.fromisoformat(result["started_at"]).timestamp()
    completed = datetime.fromisoformat(result["completed_at"]).timestamp()
    require(challenge["issued_unix_seconds"] <= started <= observation["observed_unix_seconds"] <= created + 1
            <= completed + 1 <= challenge["expires_unix_seconds"] + 1
            and abs(result["elapsed_seconds"] - (completed - started)) < 1, "Worker lifetime differs from challenge")
    for name in ("application-before.json", "application-after.json"):
        if name not in result["evidence_files"] and not result["passed"]:
            continue
        receipt = json.loads((target / name).read_bytes())
        require("shutdown" not in receipt and receipt["start_token"] == token
                and receipt["device_id"] == challenge["device_id"]
                and receipt["individual_id"] == challenge["individual_id"]
                and receipt["release_id"] == challenge["release_id"]
                and receipt["inference"]["pid"] == run["server_pid"]
                and (not result["passed"] or result["inference_pid"] == run["server_pid"]),
                "Observed application incarnation differs")
        if expected_runtime is not None:
            command = receipt["inference"]["command"]
            require(command[0] == expected_runtime["runtime_path"] and command.count("--model") == 1
                    and command[command.index("--model") + 1] == expected_runtime["model_path"]
                    and not any(arg.startswith("--lora") for arg in command)
                    and receipt["inference"]["trained_for_organism"] is True,
                    "Trained smoke inference used a different model, runtime or adapter")
    for call in job["record"]["trace"]["http_requests"]:
        require(call["body"]["model"] == "organism-" + token, "Diagnostic used a different model incarnation")
    semantic = audit_application(directory)
    require(semantic["task_successes"] == int(result["passed"]) and semantic["task_trials"] == 1,
            "Independent diagnostic evaluation disagrees with the worker")
    return {"scope": "Independent public Mac smoke-artifact readback; no receiving admission or native qualification",
            "trace_consistency_passed": True, "carried_checker_source_verified": True,
            "challenge_and_incarnation_bindings_verified": True, "independent_diagnostic_passed": bool(result["passed"]),
            "checker_seconds": result["elapsed_seconds"], "model_and_checker_exited": True,
            "native_execution_verified": False, "receiver_readiness_acknowledged": False, "qualification_passed": False,
            "limitations": ["Runtime smoke checks are public and separate from sealed quality trials",
                           "Application-owned evidence does not provide attestation against a compromised application account",
                           "Receiver acceptance still requires a separately supervised checker and current ownership verification"],
            "auditor_sha256": sha(Path(__file__).read_bytes()), "semantic_auditor_sha256": semantic["auditor_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = inspect(args.directory)
    with (args.directory / "readiness-audit.json").open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
