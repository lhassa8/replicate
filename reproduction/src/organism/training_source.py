"""Read tensor headers and verify the pinned Granite architecture without weights."""

import hashlib
import json
import math
from pathlib import Path
import struct

from .training_inputs import load, require, sha256


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate tensor-header field")
        result[key] = value
    return result


def read_header(path):
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Missing tensor-header length")
        size = struct.unpack("<Q", prefix)[0]
        require(2 <= size <= 1024 * 1024, "Tensor header exceeds its bound")
        raw = stream.read(size)
        require(len(raw) == size, "Truncated tensor header")
    value = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(value, dict), "Invalid tensor header")
    return prefix + raw, {key: entry for key, entry in value.items() if key != "__metadata__"}


def expected_shapes(config):
    require(config["model_type"] == "granite" and config["tie_word_embeddings"] is False
            and config["attention_bias"] is False and config["mlp_bias"] is False, "Unsupported source architecture")
    h, intermediate = config["hidden_size"], config["intermediate_size"]
    head = h // config["num_attention_heads"]
    require(head * config["num_attention_heads"] == h, "Fractional attention head")
    kv = head * config["num_key_value_heads"]
    shapes = {"model.embed_tokens.weight": [config["vocab_size"], h],
              "model.norm.weight": [h], "lm_head.weight": [config["vocab_size"], h]}
    projections = {"self_attn.q_proj": [h, h], "self_attn.k_proj": [kv, h],
                   "self_attn.v_proj": [kv, h], "self_attn.o_proj": [h, h],
                   "mlp.gate_proj": [intermediate, h], "mlp.up_proj": [intermediate, h],
                   "mlp.down_proj": [h, intermediate], "input_layernorm": [h], "post_attention_layernorm": [h]}
    for layer in range(config["num_hidden_layers"]):
        for name, shape in projections.items():
            shapes[f"model.layers.{layer}.{name}.weight"] = shape
    return shapes


def inspect_headers(metadata, index, headers, config, expected_parameters, *, require_matching_index=True):
    require(metadata["schema"] == "organism-native-training-source-v1" and metadata.get("failure") is None,
            "Incomplete pinned source metadata")
    weight_map = index["weight_map"]
    require(set(headers) == set(weight_map.values()), "Tensor index shard inventory differs")
    shapes, shards, observed_map, parameters, data_bytes = {}, {}, {}, 0, 0
    mismatches = []
    for name, path in headers.items():
        raw, tensors = read_header(path)
        intervals = []
        for tensor, entry in tensors.items():
            require(tensor not in shapes and tensor in weight_map, "Missing or repeated tensor")
            observed_map[tensor] = name
            if weight_map[tensor] != name:
                mismatches.append({"tensor": tensor, "index": weight_map[tensor], "header": name})
            require(set(entry) == {"dtype", "shape", "data_offsets"} and entry["dtype"] == "BF16", "Unexpected tensor dtype or fields")
            shape, offsets = entry["shape"], entry["data_offsets"]
            require(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape), "Invalid tensor shape")
            require(isinstance(offsets, list) and len(offsets) == 2 and all(type(n) is int and n >= 0 for n in offsets),
                    "Invalid tensor byte range")
            count = math.prod(shape)
            require(offsets[1] - offsets[0] == 2 * count, "Tensor shape and byte range differ")
            intervals.append(tuple(offsets))
            shapes[tensor] = shape
            parameters += count
        end = 0
        for begin, next_end in sorted(intervals):
            require(begin == end, "Tensor data has a gap or overlap")
            end = next_end
        require(len(raw) + end == metadata["files"][name]["bytes"], "Header and tensor bytes do not match the pinned shard size")
        data_bytes += end
        shards[name] = {"header_sha256": hashlib.sha256(raw).hexdigest(), "header_bytes": len(raw),
                        "tensor_bytes": end, "tensors": len(tensors)}
    require(set(shapes) == set(weight_map) and shapes == expected_shapes(config), "Native tensors differ from the model architecture")
    require(parameters == expected_parameters, "Native tensor parameter count differs")
    size_matches = index["metadata"]["total_size"] == data_bytes
    if require_matching_index:
        require(not mismatches and size_matches, "Index locations or reported size differ from actual tensor headers")
    return {"passed": not mismatches and size_matches, "parameters": parameters, "tensor_count": len(shapes), "dtype": "BF16",
            "tensor_payload_bytes": data_bytes, "native_file_bytes": sum(metadata["files"][name]["bytes"] for name in headers),
            "index_reported_tensor_bytes": index["metadata"]["total_size"],
            "index_reported_size_matches_headers": size_matches, "index_placement_mismatches": mismatches,
            "observed_weight_map": observed_map, "shards": shards, "full_native_payload_verified": False}


def inspect_recipe_source(paths, expected_parameters):
    metadata, index, config = load(paths["base_metadata"]), load(paths["base_index"]), load(paths["base_config"])
    headers = {name.removeprefix("base_header/"): path for name, path in paths.items() if name.startswith("base_header/")}
    original = inspect_headers(metadata, index, headers, config, expected_parameters, require_matching_index=False)
    repaired = load(paths["loader_index"])
    receipt = load(paths["index_repair_receipt"])
    require(receipt["upstream_metadata_sha256"] == sha256(paths["base_metadata"])
            and receipt["upstream_index_sha256"] == sha256(paths["base_index"])
            and receipt["derived_loader_index_sha256"] == sha256(paths["loader_index"]),
            "Loader-index derivation does not bind the pinned source")
    require(repaired == {"metadata": {"total_size": original["tensor_payload_bytes"]},
                         "weight_map": original["observed_weight_map"]}, "Derived index does not exactly describe the headers")
    checked = inspect_headers(metadata, repaired, headers, config, expected_parameters)
    require(checked["passed"], "Derived loader index failed strict verification")
    checked["upstream_index_consistent"] = original["passed"]
    checked["upstream_index_placement_mismatches"] = len(original["index_placement_mismatches"])
    checked["upstream_index_reported_tensor_bytes"] = original["index_reported_tensor_bytes"]
    checked["derived_loader_index_sha256"] = sha256(paths["loader_index"])
    return checked
