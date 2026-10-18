"""Load the pinned BF16 base in Mac RAM and check a bounded CPU forward pass."""

from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import resource
import signal
import time
import urllib.request

from .training_inputs import (check_versions, load, offline_environment, require,
                              sha256, validate_recipe, verified_path)
from .training_source import expected_shapes, inspect_recipe_source
from .verify_training_stream import FILES, HttpsRedirects, stream_file


MAXIMUM_SECONDS = 1800
PREFIX_TOKENS = 32


class CapturingResponse:
    """Keep only this bounded file in memory, using the existing stream checks."""

    def __init__(self, response, maximum):
        self.response, self.maximum = response, maximum
        self.buffer = io.BytesIO()
        self.bytes = 0
        self.status, self.headers = response.status, response.headers

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def geturl(self):
        return self.response.geturl()

    def read(self, maximum):
        block = self.response.read(maximum)
        self.bytes += len(block)
        require(self.bytes <= self.maximum, "Memory capture exceeded its pinned file size")
        self.buffer.write(block)
        return block


def memory_download(opener, entry, prefix, deadline, state, persist):
    require(type(entry["bytes"]) is int and 0 < entry["bytes"] <= 6 * 1024**3,
            "Memory download exceeds the file bound")
    response = None

    class Capture:
        def open(self, request, timeout):
            nonlocal response
            require(response is None, "A memory download may open only one response")
            response = CapturingResponse(opener.open(request, timeout=timeout), entry["bytes"])
            return response

    try:
        stream_file(Capture(), entry, prefix, deadline, state, persist)
        require(response is not None and state["full_payload_verified"], "Unverified source cannot be loaded")
        return response.buffer.getvalue()
    finally:
        if response is not None:
            response.buffer.close()


def empty_model(config):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM, GraniteConfig
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            GraniteConfig(**config), dtype=torch.bfloat16, attn_implementation="sdpa")
    require(type(model).__name__ == "GraniteForCausalLM", "Unexpected model architecture")
    require({name: list(value.shape) for name, value in model.named_parameters()} == expected_shapes(config),
            "Empty model differs from the pinned tensor shapes")
    require(all(value.device.type == "meta" and value.dtype == torch.bfloat16 for value in model.parameters()),
            "Empty model allocated or changed its base parameters")
    require(all(value.device.type == "cpu" for value in model.buffers()), "A required buffer is not materialized on CPU")
    return model


def assign_blob(model, blob, expected_entry, expected_names, loaded):
    """Use the installed safetensors loader; partial assignment has exact checks."""
    import torch
    from safetensors.torch import load as load_tensors
    require(len(blob) == expected_entry["bytes"] and hashlib.sha256(blob).hexdigest() == expected_entry["sha256"],
            "Memory blob differs from the fully verified source")
    require(expected_names and not set(expected_names) & set(loaded), "Repeated or empty source shard")
    tensors = load_tensors(blob)
    require(set(tensors) == set(expected_names), "Decoded shard tensor inventory differs")
    shapes = {name: list(value.shape) for name, value in model.named_parameters()}
    require(all(name in shapes and list(value.shape) == shapes[name]
                and value.dtype == torch.bfloat16 and value.device.type == "cpu"
                for name, value in tensors.items()), "Decoded source tensor shape, dtype or device differs")
    result = model.load_state_dict(tensors, strict=False, assign=True)
    expected_remaining = set(shapes) - set(loaded) - set(expected_names)
    # PyTorch reports keys absent from this call, including earlier shards.
    # Actual meta parameters independently identify what remains unloaded.
    require(not result.unexpected_keys and set(result.missing_keys) == set(shapes) - set(expected_names),
            "Partial model loading left unexplained keys")
    require({name for name, value in model.named_parameters() if value.device.type == "meta"} == expected_remaining,
            "Materialized parameters disagree with the complete loading history")
    loaded.update(expected_names)
    return {"loaded_tensors": len(expected_names), "remaining_tensors": len(expected_remaining)}


def tensor_hashes(model):
    import torch
    result = {}
    for name, value in model.named_parameters():
        require(value.device.type == "cpu" and value.dtype == torch.bfloat16
                and value.is_contiguous(), "Unexpected materialized parameter")
        raw = value.detach().view(torch.uint8).numpy()
        result[name] = hashlib.sha256(memoryview(raw)).hexdigest()
    return result


def forward_check(model, token_ids):
    import torch
    require(isinstance(token_ids, list) and 2 <= len(token_ids) <= PREFIX_TOKENS,
            "CPU forward input exceeds the small diagnostic bound")
    require(all(type(token) is int and 0 <= token < model.config.vocab_size for token in token_ids),
            "CPU forward input contains an invalid token")
    model.eval()
    model.requires_grad_(False)
    before = tensor_hashes(model)
    require(all(value.device.type == "cpu" and torch.isfinite(value).all().item()
                for value in model.buffers()), "A model buffer is absent or nonfinite")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    started = time.monotonic()
    with torch.inference_mode():
        first = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        first_seconds = time.monotonic() - started
        require(tuple(first.shape) == (1, len(token_ids), model.config.vocab_size)
                and torch.isfinite(first).all().item(), "Full-model logits have an invalid shape or nonfinite value")
        second = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        two_pass_seconds = time.monotonic() - started
        require(torch.equal(first, second), "Repeated CPU forward passes disagree")
        loss = torch.nn.functional.cross_entropy(first[:, :-1].float().reshape(-1, first.shape[-1]),
                                                input_ids[:, 1:].reshape(-1))
        require(torch.isfinite(loss).item(), "Development-prefix next-token loss is nonfinite")
    after = tensor_hashes(model)
    require(after == before, "A CPU forward pass changed the source parameters")
    require(all(value.grad is None and not value.requires_grad for value in model.parameters()),
            "CPU verification enabled a gradient")
    raw_logits = first.float().contiguous().numpy()
    return {
        "passed": True, "input_token_ids": token_ids, "sequence_tokens": len(token_ids),
        "logits_shape": list(first.shape), "logits_dtype": str(first.dtype),
        "logits_float32_sha256": hashlib.sha256(memoryview(raw_logits)).hexdigest(),
        "next_token_id": int(first[0, -1].argmax().item()),
        "development_prefix_next_token_loss": loss.item(),
        "first_forward_seconds": round(first_seconds, 3),
        "two_forward_passes_seconds": round(two_pass_seconds, 3),
        "repeated_logits_identical": True, "parameter_hashes_before": before,
        "parameter_hashes_after": after, "parameters_unchanged": True,
        "optimizer_steps": 0, "backward_pass_executed": False,
        "quality_evaluation": False, "full_4096_token_fit_verified": False,
    }


def run(root, output):
    root, output = Path(root).resolve(), Path(output)
    require(platform.system() == "Darwin" and platform.machine() == "arm64",
            "This development probe is limited to the declared Mac")
    offline_environment()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": "organism-native-memory-probe-v1", "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "checking_inputs", "platform": platform.platform(), "failure": None,
        "maximum_seconds": MAXIMUM_SECONDS, "cpu_threads": 4, "prefix_token_limit": PREFIX_TOKENS,
        "native_model_loaded_on_mac": False, "native_linux_execution_verified": False,
        "cuda_verified": False, "large_weight_files_saved": False,
        "optimizer_steps": 0, "training_completed": False, "qualification_passed": False,
        "files": {},
    }

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        temporary = output / ".report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "report.json")

    def check_deadline():
        require(time.monotonic() - started < MAXIMUM_SECONDS, "Memory-probe deadline expired")

    def stopped(signum, frame):
        raise TimeoutError("The supervising process requested this probe to stop")

    signal.signal(signal.SIGTERM, stopped)
    persist()
    try:
        recipe_path = root / "organism/training-recipe.draft.json"
        recipe = load(recipe_path)
        validate_recipe(recipe)
        paths = {name: verified_path(root, entry) for name, entry in recipe["inputs"].items()}
        metadata, config = load(paths["base_metadata"]), load(paths["base_config"])
        require(set(metadata["files"]) == FILES and metadata["repository"] == recipe["base"]["repository"]
                and metadata["revision"] == recipe["base"]["revision"], "Pinned native source differs")
        report["recipe_sha256"] = sha256(recipe_path)
        report["metadata_sha256"] = sha256(paths["base_metadata"])
        report["header_source"] = inspect_recipe_source(paths, recipe["base"]["expected_parameters"])
        report["versions"] = check_versions()
        report["source_hashes"] = {name: sha256(Path(__file__).with_name(name)) for name in (
            "verify_training_memory.py", "verify_training_stream.py", "training_inputs.py", "training_source.py")}
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(recipe["training"]["seed"])
        model = empty_model(config)
        expected_parameters = recipe["base"]["expected_parameters"]
        require(sum(value.numel() for value in model.parameters()) == expected_parameters,
                "Empty model parameter count differs")
        loader_index = load(paths["loader_index"])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), HttpsRedirects())
        loaded = set()
        for name, entry in metadata["files"].items():
            check_deadline()
            expected_url = ("https://huggingface.co/" + metadata["repository"] + "/resolve/"
                            + metadata["revision"] + "/" + name)
            require(entry["source_url"] == expected_url, "Memory source URL differs from the pinned revision")
            prefix_path = (paths["base_header/" + name] if name.endswith(".safetensors")
                           else paths["base_config" if name == "config.json" else "base_index"])
            state = {"expected_bytes": entry["bytes"], "expected_sha256": entry["sha256"]}
            report["files"][name] = state
            report["status"] = "downloading_verified_source"
            persist()
            blob = memory_download(opener, entry, prefix_path.read_bytes(), started + MAXIMUM_SECONDS, state, persist)
            if name.endswith(".safetensors"):
                report["status"] = "loading_verified_shard"
                persist()
                names = {key for key, value in loader_index["weight_map"].items() if value == name}
                state["assignment"] = assign_blob(model, blob, entry, names, loaded)
            del blob
            gc.collect()
            persist()
        check_deadline()
        require(loaded == set(expected_shapes(config))
                and all(value.device.type == "cpu" and value.dtype == torch.bfloat16 for value in model.parameters()),
                "The complete native model was not materialized on CPU")
        report["native_model_loaded_on_mac"] = True
        report["base_parameters"] = sum(value.numel() for value in model.parameters())
        report["loaded_tensors"] = len(loaded)
        require(report["base_parameters"] == expected_parameters, "Materialized parameter count differs")
        with paths["tokens_train"].open() as stream:
            first_row = json.loads(stream.readline())
        require(first_row["prompt_tokens"] >= PREFIX_TOKENS, "Development prefix is shorter than the probe")
        report["development_input"] = {"case_id": first_row["case_id"], "decision_index": first_row["decision_index"],
                                       "source_path": recipe["inputs"]["tokens_train"]["path"],
                                       "source_sha256": sha256(paths["tokens_train"]), "prefix_tokens": PREFIX_TOKENS}
        report["status"] = "checking_cpu_forward"
        persist()
        report["forward_check"] = forward_check(model, first_row["input_ids"][:PREFIX_TOKENS])
        check_deadline()
        report["status"] = "verified_mac_cpu_not_qualified"
        report["passed"] = True
        report["limitations"] = [
            "Official safetensors byte loading plus exact partial state assignment was used, not the GPU worker's file-based from_pretrained path.",
            "The two 32-token CPU forward passes do not test a complete task, backward pass, training, CUDA, Linux or 4096-token fit.",
            "The development Mac has 32 GiB; this does not establish the 8 GiB deployment profile.",
        ]
        del model
        gc.collect()
    except BaseException as exc:
        report["status"] = "failed"
        report["passed"] = False
        report["failure"] = {"type": type(exc).__name__,
                             "message": re.sub(r"https?://\S+", "<url>", str(exc))[:1000]}
        raise
    finally:
        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        persist()
    return report
