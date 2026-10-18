"""Run with organism/.venv-training: small real-library memory-loading checks."""

import hashlib
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save
from transformers import AutoModelForCausalLM, GraniteConfig

from .verify_training_memory import assign_blob, empty_model, forward_check, tensor_hashes


def identity(blob):
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def fixture():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = GraniteConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, pad_token_id=2, eos_token_id=2,
                           max_position_embeddings=64, use_cache=False, attention_dropout=0.0)
    reference = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    state = reference.state_dict()
    names = sorted(state)
    split = len(names) // 2
    shards = [{name: state[name] for name in group} for group in (names[:split], names[split:])]
    return reference, config.to_dict(), shards


class MaterializationTests(unittest.TestCase):
    def test_two_real_safetensors_blobs_restore_all_values_and_forward_output(self):
        reference, config, shards = fixture()
        restored = empty_model(config)
        self.assertTrue(all(p.device.type == "meta" for p in restored.parameters()))
        loaded = set()
        for shard in shards:
            blob = save(shard)
            result = assign_blob(restored, blob, identity(blob), set(shard), loaded)
        self.assertEqual(result["remaining_tensors"], 0)
        self.assertEqual(tensor_hashes(reference), tensor_hashes(restored))
        tokens = [5, 6, 7, 8, 9, 2]
        expected = forward_check(reference, tokens)
        actual = forward_check(restored, tokens)
        self.assertEqual(actual["logits_float32_sha256"], expected["logits_float32_sha256"])
        self.assertTrue(actual["parameters_unchanged"])
        self.assertEqual(actual["optimizer_steps"], 0)

    def test_corrupt_blob_rejected_before_decoder_or_model_assignment(self):
        _, config, shards = fixture()
        model = empty_model(config)
        blob = save(shards[0])
        corrupt = blob[:-1] + bytes([blob[-1] ^ 1])
        with self.assertRaisesRegex(ValueError, "fully verified"):
            assign_blob(model, corrupt, identity(blob), set(shards[0]), set())
        self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))

    def test_wrong_inventory_dtype_and_shape_rejected_before_assignment(self):
        _, config, shards = fixture()
        for mutation in ("inventory", "dtype", "shape"):
            model = empty_model(config)
            shard = dict(shards[0])
            names = set(shard)
            key = next(iter(shard))
            if mutation == "inventory":
                names = {"undeclared.weight"}
            elif mutation == "dtype":
                shard[key] = shard[key].float()
            else:
                shard[key] = shard[key].reshape(-1)[:1]
            blob = save(shard)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                assign_blob(model, blob, identity(blob), names, set())
            self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))

    def test_duplicate_shard_is_rejected(self):
        _, config, shards = fixture()
        model, loaded = empty_model(config), set()
        blob = save(shards[0])
        assign_blob(model, blob, identity(blob), set(shards[0]), loaded)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            assign_blob(model, blob, identity(blob), set(shards[0]), loaded)

    def test_nonfinite_logits_are_not_reported_as_verified(self):
        model, _, _ = fixture()
        with torch.no_grad():
            model.lm_head.weight[0].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            forward_check(model, [5, 6, 7, 8])

    def test_unused_weight_mutation_is_detected_even_when_logits_match(self):
        model, _, _ = fixture()
        original = model.forward
        def mutated(*args, **kwargs):
            result = original(*args, **kwargs)
            model.model.embed_tokens.weight[50, 0].add_(1)
            return result
        with patch.object(model, "forward", side_effect=mutated), self.assertRaisesRegex(ValueError, "changed"):
            forward_check(model, [5, 6, 7, 8])

    def test_forward_token_limit_is_enforced(self):
        model, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "bound"):
            forward_check(model, [5] * 33)


if __name__ == "__main__":
    unittest.main()
