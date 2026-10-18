"""Explicit Q4_K_M mixtures; the default and prior profile remain unchanged."""

import re
from .training_inputs import require


PROFILE = "q4-k-m-io-q8-v1"
BODY_PROFILE = "q4-k-m-io-q8-down-v-q6-v1"
BODY_PATTERN = r"^blk\.[0-9]+\.(attn_v|ffn_down)\.weight$"
ATTENTION_PROFILE = "q4-k-m-io-q8-attn-down-q6-v1"
ATTENTION_PATTERN = r"^blk\.[0-9]+\.(attn_q|attn_k|attn_v|attn_output|ffn_down)\.weight$"


def specification(profile):
    require(profile in (PROFILE, BODY_PROFILE, ATTENTION_PROFILE), "Unsupported quantization precision profile")
    result = {
        "schema": "organism-quantization-precision-v1", "profile": profile,
        "family": "Q4_K_M",
        "overrides": {"output.weight": "Q8_0", "token_embd.weight": "Q8_0"},
    }
    if profile != PROFILE:
        result["schema"] = "organism-quantization-precision-v2"
        result["pattern_overrides"] = {
            BODY_PATTERN if profile == BODY_PROFILE else ATTENTION_PATTERN: "Q6_K"}
    return result


def validate_specification(value):
    require(isinstance(value, dict) and value == specification(value.get("profile")),
            "Quantization precision specification changed")
    return value["profile"]


def arguments(profile):
    specification(profile)
    result = ["--output-tensor-type", "q8_0", "--token-embedding-type", "q8_0"]
    if profile != PROFILE:
        pattern = BODY_PATTERN if profile == BODY_PROFILE else ATTENTION_PATTERN
        result.extend(["--tensor-type", pattern + "=q6_k"])
    return result


def inspect_precision(path, profile):
    """Read actual tensor types; a family label alone cannot prove the mixture."""
    import gguf

    spec = specification(profile)
    reader = gguf.GGUFReader(path)
    require(reader.fields["general.file_type"].contents() == int(gguf.LlamaFileType.MOSTLY_Q4_K_M),
            "Precision profile requires the Q4_K_M family")
    tensors = {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}
    require(len(tensors) == len(reader.tensors)
            and all(tensors.get(name) == kind for name, kind in spec["overrides"].items()),
            "GGUF omitted or changed an explicit precision override")
    body_weights = 0
    overridden = set()
    for tensor in reader.tensors:
        if tensor.name in spec["overrides"]:
            require(len(tensor.shape) == 2, "Precision override is not a matrix")
        elif len(tensor.shape) == 1:
            require(tensor.name.endswith("_norm.weight") and tensor.tensor_type.name == "F32",
                    "Unexpected scalar or normalization precision")
        else:
            require(len(tensor.shape) == 2 and tensor.name.startswith("blk.")
                    and tensor.tensor_type.name in ("Q4_K", "Q6_K"),
                    "Unexpected body tensor precision")
            if profile != PROFILE:
                require(re.fullmatch(r"blk\.[0-9]+\.(attn_k|attn_output|attn_q|attn_v|ffn_down|ffn_gate|ffn_up)\.weight",
                                     tensor.name), "Unexpected body weight in the explicit profile")
                pattern = BODY_PATTERN if profile == BODY_PROFILE else ATTENTION_PATTERN
                selected = re.fullmatch(pattern, tensor.name) is not None
                require(tensor.tensor_type.name == ("Q6_K" if selected else "Q4_K"),
                        "Explicit body precision differs")
                if selected:
                    overridden.add(tensor.name)
            body_weights += 1
    require(body_weights > 0, "Precision profile has no quantized body")
    if profile != PROFILE:
        field = reader.fields.get("granite.block_count")
        layers = field.contents() if field is not None else None
        require(type(layers) is int and 1 <= layers <= 128,
                "Explicit body profile requires its Granite layer count")
        roles = (("attn_v", "ffn_down") if profile == BODY_PROFILE
                 else ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_down"))
        require(overridden == {f"blk.{layer}.{role}.weight" for layer in range(layers) for role in roles},
                "Explicit body profile omitted an override")
    return {"specification": spec, "tensor_types_by_name": tensors}
