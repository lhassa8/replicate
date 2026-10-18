"""Bound local GGUF conversion to recorded tools, tensor inventory and limits."""

from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import tarfile
import time

from .training_inputs import check_versions, require, sha256, verified_path
from .runtime_environment import library_environment


COMMIT = "4f31eedb0ccf546b7e8d6bb243b170f12522f54d"
SOURCE_ARCHIVE_SHA256 = "537e725e64f1457874a8d9d984575b203471e1b0a476ad5eee481b057fc0df54"

CONVERTER_WRAPPER = """
import pathlib, runpy, sys
dependency, script = (pathlib.Path(value).resolve() for value in sys.argv[1:3])
sys.argv = sys.argv[2:]
sys.path.insert(0, str(dependency))
sys.path.insert(0, str(script.parent))
def guard(event, arguments):
    # Linux ctypes uses this read-only cache query while importing PyTorch.
    # Keep the exact executable, arguments and inherited working directory.
    if (sys.platform == 'linux' and event == 'subprocess.Popen'
            and len(arguments) == 4 and arguments[0] == '/sbin/ldconfig'
            and arguments[1] == ['/sbin/ldconfig', '-p'] and arguments[2] is None
            and arguments[3] == {'LC_ALL': 'C', 'LANG': 'C'}):
        return
    if event.startswith(('socket.', 'subprocess.', 'os.exec', 'os.posix_spawn')) or event == 'os.system':
        raise RuntimeError('Converter network/process operation is not allowed: ' + event)
sys.addaudithook(guard)
runpy.run_path(str(script), run_name='__main__')
"""


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def bind_toolchain(prior, *, source_root, source_archive, converter, quantizer, server,
                   dependency_root, library_paths=(), symlinks=None, system=None, machine=None):
    """Bind an already recorded input inventory; this does not install tools."""
    return {
        "schema": "organism-gguf-export-tools-v1", "declared_commit": COMMIT,
        "system": system or platform.system(), "machine": machine or platform.machine(),
        "source_root": source_root, "source_archive": {"path": source_archive, **prior["source_archive"]},
        "converter": converter, "quantizer": quantizer, "server": server,
        "dependency_root": dependency_root, "library_paths": list(library_paths),
        "files": prior["files"], "symlinks": symlinks or {},
        "prior_input_receipt_sha256": hashlib.sha256(canonical(prior)).hexdigest(),
        "qualification_passed": False,
    }


@dataclass(frozen=True)
class Toolchain:
    root: Path
    manifest: dict

    def path(self, name):
        value = Path(name)
        require(not value.is_absolute() and value.parts and ".." not in value.parts,
                "Export tool path escapes its root")
        path = self.root / value
        require(path.resolve().is_relative_to(self.root.resolve()), "Export tool resolves outside its root")
        return path

    def verify(self):
        m = self.manifest
        require(m.get("schema") == "organism-gguf-export-tools-v1" and m.get("declared_commit") == COMMIT,
                "Unrecognized export toolchain")
        require((m["system"], m["machine"]) == (platform.system(), platform.machine()),
                "Export tools belong to a different host platform")
        require((m["system"], m["machine"]) in (("Darwin", "arm64"), ("Linux", "x86_64"),
                                               ("Linux", "aarch64")), "Unsupported export host profile")
        require(m["source_archive"]["sha256"] == SOURCE_ARCHIVE_SHA256, "Converter source archive is not pinned")
        archive = verified_path(self.root, m["source_archive"])
        require(isinstance(m["files"], dict) and 3 <= len(m["files"]) <= 1024, "Invalid export tool inventory")
        require(isinstance(m["symlinks"], dict) and len(m["symlinks"]) <= 128
                and set(m["symlinks"]).issubset(m["files"]),
                "Invalid export library alias inventory")
        for name, entry in m["files"].items():
            require(set(entry) == {"bytes", "sha256"}, "Unexpected tool identity fields")
            if name in m["symlinks"]:
                require(identity(self.path(name)) == entry, "Export library alias target changed")
            else:
                verified_path(self.root, {"path": name, **entry})
        for name, target in m["symlinks"].items():
            path = self.path(name)
            require(path.is_symlink() and os.readlink(path) == target
                    and str(path.resolve().relative_to(self.root.resolve())) in m["files"],
                    "Unbound export library alias")
        for role in ("converter", "quantizer", "server"):
            require(m[role] in m["files"], "An executable is absent from the tool inventory")
        if "imatrix" in m:
            require(m["imatrix"] in m["files"] and m["imatrix"] not in m["symlinks"]
                    and self.path(m["imatrix"]).name == "llama-imatrix",
                    "Calibration executable is absent or aliased")
        require(self.path(m["converter"]).name == "convert_hf_to_gguf.py"
                and self.path(m["quantizer"]).name == "llama-quantize"
                and self.path(m["server"]).name == "llama-server", "Unexpected export executable")
        source = self.path(m["source_root"])
        require(self.path(m["converter"]) == source / "convert_hf_to_gguf.py",
                "Selected converter is outside the pinned source tree")
        expected_sources = [source / "convert_hf_to_gguf.py"]
        expected_sources += list((source / "conversion").rglob("*.py"))
        expected_sources += list((source / "gguf-py/gguf").rglob("*.py"))
        require(len(expected_sources) > 10, "Converter source inventory is incomplete")
        with tarfile.open(archive, "r:gz") as original:
            for path in expected_sources:
                relative = str(path.relative_to(self.root))
                require(relative in m["files"], "Converter imports unbound source")
                member = original.getmember("llama.cpp-" + COMMIT + "/" + str(path.relative_to(source)))
                require(member.isfile() and member.size <= 2 * 1024**2
                        and original.extractfile(member).read() == path.read_bytes(),
                        "Converter source differs from the pinned archive")
        dependency = self.path(m["dependency_root"])
        require(str((dependency / "sentencepiece/__init__.py").relative_to(self.root)) in m["files"],
                "Pinned tokenizer dependency is missing")
        for directory in [self.path(m["server"]).parent, *[self.path(name) for name in m["library_paths"]]]:
            require(directory.is_dir(), "Export library directory is missing")
            for path in directory.iterdir():
                if ".so" in path.name or path.suffix == ".dylib":
                    name = str(path.relative_to(self.root))
                    require(name in m["files"] or name in m["symlinks"], "An export library is not bound")
        check_versions()
        require(version("numpy") == "2.5.4", "Export NumPy version changed")
        return {"manifest_sha256": hashlib.sha256(canonical(m)).hexdigest(),
                "verified_files": len(m["files"]), "profile": [m["system"], m["machine"]],
                "native_host_execution_qualified": False}

    def import_gguf(self):
        source = self.path(self.manifest["source_root"]) / "gguf-py"
        sys.path.insert(0, str(source))
        import gguf
        require(Path(gguf.__file__).resolve().is_relative_to(source), "Unexpected GGUF library origin")
        return gguf


def run_command(argv, log_path, records, *, timeout_seconds, library_paths=(),
                output_path=None, maximum_output_bytes=20 * 1024**3):
    require(0 < timeout_seconds <= 1200, "Invalid conversion command deadline")
    require(type(maximum_output_bytes) is int and 0 < maximum_output_bytes <= 20 * 1024**3,
            "Invalid conversion output allowance")
    environment = {key: value for key, value in os.environ.items() if key in ("PATH", "LANG", "TMPDIR")}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4",
                       AWS_EC2_METADATA_DISABLED="true", OTEL_SDK_DISABLED="true")
    environment.update(library_environment(library_paths))
    began = time.monotonic()
    item = {"argv": list(map(str, argv)), "log": log_path.name, "timeout_seconds": timeout_seconds,
            "maximum_log_bytes": 16 * 1024**2, "maximum_output_bytes": maximum_output_bytes}
    records.append(item)
    with log_path.open("xb") as log:
        process = None
        try:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       start_new_session=True)
            while True:
                require(log_path.stat().st_size <= item["maximum_log_bytes"], "Conversion log exceeded its bound")
                if output_path is not None and output_path.exists():
                    require(output_path.stat().st_size <= maximum_output_bytes, "Conversion output exceeded its bound")
                if process.poll() is not None:
                    break
                if time.monotonic() - began > timeout_seconds:
                    item["timed_out"] = True
                    raise TimeoutError("Conversion command exhausted its deadline")
                time.sleep(0.1)
            item["return_code"] = process.returncode
        except BaseException as exc:
            item["failure"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            if process is not None:
                item["return_code"] = process.returncode
            item["seconds"] = round(time.monotonic() - began, 3)
    require(process.returncode == 0, "Conversion command failed; retained log: " + str(log_path))


def converter_command(tools, arguments):
    return [sys.executable, "-I", "-c", CONVERTER_WRAPPER,
            str(tools.path(tools.manifest["dependency_root"])),
            str(tools.path(tools.manifest["converter"])), *map(str, arguments)]


def convert_f32(tools, source, target, log, records, *, model_name, timeout_seconds=180):
    require(not target.exists(), "F32 export target already exists")
    run_command(converter_command(tools, [source, "--outfile", target, "--outtype", "f32",
                                         "--model-name", model_name]),
                log, records, timeout_seconds=timeout_seconds, output_path=target,
                library_paths=tuple(tools.path(name) for name in tools.manifest["library_paths"]))


def quantize_q4(tools, source, target, log, records, *, timeout_seconds=180, imatrix=None,
                precision_profile=None):
    require(not target.exists(), "Quantized export target already exists")
    extra = []
    if imatrix is not None:
        require(imatrix.is_file() and not imatrix.is_symlink()
                and 0 < imatrix.stat().st_size <= 16 * 1024**2, "Invalid importance matrix input")
        extra = ["--imatrix", str(imatrix)]
    if precision_profile is not None:
        from .quantization_precision import arguments
        extra += arguments(precision_profile)
    run_command([str(tools.path(tools.manifest["quantizer"])), *extra, str(source), str(target), "Q4_K_M", "4"],
                log, records, timeout_seconds=timeout_seconds,
                library_paths=tuple(tools.path(name) for name in tools.manifest["library_paths"]),
                output_path=target)


def tokenizer_fingerprint(path):
    import gguf
    reader = gguf.GGUFReader(path)
    return {key: hashlib.sha256(json.dumps(field.contents(), ensure_ascii=True, allow_nan=False,
                                           separators=(",", ":")).encode()).hexdigest()
            for key, field in reader.fields.items() if key.startswith("tokenizer.")}


def inspect_gguf(path, config, tokenizer_reference):
    import gguf
    reader = gguf.GGUFReader(path)
    size, feed = config["hidden_size"], config["intermediate_size"]
    expected = {"token_embd.weight": [config["vocab_size"], size],
                "output.weight": [config["vocab_size"], size], "output_norm.weight": [size]}
    head = size // config["num_attention_heads"]
    kv = head * config["num_key_value_heads"]
    for layer in range(config["num_hidden_layers"]):
        for name, shape in {
            "attn_q": [size, size], "attn_k": [kv, size], "attn_v": [kv, size],
            "attn_output": [size, size], "ffn_gate": [feed, size], "ffn_up": [feed, size],
            "ffn_down": [size, feed], "attn_norm": [size], "ffn_norm": [size],
        }.items():
            expected[f"blk.{layer}.{name}.weight"] = shape
    actual = {tensor.name: tensor.shape[::-1].tolist() for tensor in reader.tensors}
    require(len(actual) == len(reader.tensors) and actual == expected, "GGUF tensor inventory or shape changed")
    required = {"general.architecture": "granite", "granite.block_count": config["num_hidden_layers"],
                "granite.embedding_length": size, "granite.context_length": config["max_position_embeddings"],
                "granite.feed_forward_length": feed, "granite.vocab_size": config["vocab_size"],
                "granite.rope.dimension_count": head,
                "granite.attention.head_count": config["num_attention_heads"],
                "granite.attention.head_count_kv": config["num_key_value_heads"]}
    floats = {"granite.rope.freq_base": config["rope_parameters"]["rope_theta"],
              "granite.attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
              "granite.attention.scale": config["attention_multiplier"],
              "granite.embedding_scale": config["embedding_multiplier"],
              "granite.residual_scale": config["residual_multiplier"],
              "granite.logit_scale": config["logits_scaling"]}
    required.update({key: struct.unpack("<f", struct.pack("<f", value))[0] for key, value in floats.items()})
    require(all(key in reader.fields and reader.fields[key].contents() == value
                for key, value in required.items()), "GGUF computational metadata differs")
    require(tokenizer_fingerprint(path) == tokenizer_reference, "GGUF tokenizer identity differs")
    return {**identity(path), "tensor_count": len(reader.tensors),
            "parameters": sum(tensor.n_elements for tensor in reader.tensors),
            "tensor_types": dict(Counter(tensor.tensor_type.name for tensor in reader.tensors)),
            "file_type": reader.fields["general.file_type"].contents(),
            "tokenizer_fields_identical": len(tokenizer_reference), "architecture": "granite"}
