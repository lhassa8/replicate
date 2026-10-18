"""Run with organism/.venv-training/bin/python -m organism.training_tests."""

import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .training_inputs import ExactTargetCollator, make_recipe, offline_environment, sha256
offline_environment()
import torch
from transformers import AutoModelForCausalLM, GraniteConfig, GraniteForCausalLM
from safetensors.torch import save_file

from .training_preflight import architecture_inventory, tiny_gradient_check
from .training_source import inspect_headers
from .train_specialist import prepare_loading_view, target_token_loss, verify_native_base


ROOT = Path(__file__).resolve().parents[1]


class TrainingMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = make_recipe(ROOT)

    def test_meta_architecture_matches_selected_source_without_allocating_weights(self):
        config = json.loads((ROOT / self.recipe["inputs"]["base_config"]["path"]).read_bytes())
        result = architecture_inventory(config, self.recipe["adapter"])
        self.assertTrue(result["meta_only"])
        self.assertEqual(result["base_parameters"], 3659737600)
        self.assertEqual(result["adapter_parameters"], 31129600)
        self.assertEqual(result["target_projections"], 280)

    def test_real_eos_is_preserved_when_padding_uses_the_same_id(self):
        batch = ExactTargetCollator(2)([
            {"input_ids": [8, 2], "labels": [-100, 2]},
            {"input_ids": [9, 5, 2], "labels": [-100, 5, 2]},
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[8, 2, 2], [9, 5, 2]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2, -100], [-100, 5, 2]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 0], [1, 1, 1]])

    def test_loss_is_next_token_only_and_masked_predictions_have_zero_gradient(self):
        logits = torch.zeros(2, 5, 11, requires_grad=True)
        labels = torch.tensor([[-100, -100, 5, 3, 2], [-100, -100, -100, -100, 2]])
        loss = target_token_loss(SimpleNamespace(logits=logits), labels)
        self.assertAlmostEqual(loss.item(), math.log(11), places=6)
        loss.backward()
        masked = labels[:, 1:] == -100
        self.assertTrue(torch.equal(logits.grad[:, :-1][masked], torch.zeros_like(logits.grad[:, :-1][masked])))
        self.assertTrue(torch.equal(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1])))
        self.assertTrue(logits.grad[:, :-1][~masked].abs().sum().item() > 0)

    def test_zero_target_or_nonfinite_loss_is_rejected(self):
        outputs = SimpleNamespace(logits=torch.zeros(1, 3, 5))
        with self.assertRaisesRegex(ValueError, "no supervised"):
            target_token_loss(outputs, torch.full((1, 3), -100))
        outputs.logits[0, 1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            target_token_loss(outputs, torch.tensor([[-100, -100, 2]]))

    def test_actual_trainer_accumulation_matches_one_batch_without_optimizer_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = tiny_gradient_check(self.recipe, temporary)
        self.assertTrue(result["accumulated_gradients_match_full_batch"])
        self.assertTrue(result["parameters_unchanged"])
        self.assertEqual(result["optimizer_steps"], 0)

    def test_derived_index_loads_tiny_verified_shards_and_preserves_original_files(self):
        # An explicit tiny fixture validates the loading mechanism, not the 3B payload.
        torch.manual_seed(7)
        config = GraniteConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, use_cache=False)
        model = GraniteForCausalLM(config)
        config.architectures = ["GraniteForCausalLM"]
        state = {name: value.detach().to(torch.bfloat16).contiguous() for name, value in model.state_dict().items()}
        names = sorted(state)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            config.save_pretrained(base)
            actual_map = {}
            for number, selected in enumerate((names[::2], names[1::2]), 1):
                name = f"model-{number:05d}-of-00002.safetensors"
                save_file({key: state[key] for key in selected}, base / name, metadata={"format": "pt"})
                actual_map.update({key: name for key in selected})
            other = {name: ("model-00002-of-00002.safetensors" if shard.endswith("01-of-00002.safetensors")
                            else "model-00001-of-00002.safetensors") for name, shard in actual_map.items()}
            index_path = base / "model.safetensors.index.json"
            index_path.write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": other}))
            parameters = sum(value.numel() for value in state.values())
            derived = root / "derived.json"
            derived.write_text(json.dumps({"metadata": {"total_size": parameters * 2}, "weight_map": actual_map}))
            metadata = {"schema": "organism-native-training-source-v1", "repository": "unit-test-fixture",
                        "revision": "a" * 40, "failure": None,
                        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in base.iterdir()}}
            recipe = {"base": {"repository": metadata["repository"], "revision": metadata["revision"], "expected_parameters": parameters}}
            checked = verify_native_base(base, metadata, recipe, derived)
            self.assertTrue(checked["passed"])
            with self.assertRaisesRegex(ValueError, "Index locations"):
                inspect_headers(metadata, json.loads(index_path.read_bytes()),
                                {name: base / name for name in metadata["files"] if name.endswith(".safetensors")},
                                json.loads((base / "config.json").read_bytes()), parameters)
            view = prepare_loading_view(base, metadata, derived, root / "view")
            restored, info = AutoModelForCausalLM.from_pretrained(view, local_files_only=True, trust_remote_code=False,
                                                                use_safetensors=True, dtype=torch.bfloat16,
                                                                output_loading_info=True)
            self.assertFalse(any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")))
            self.assertTrue(all(torch.equal(value, state[name]) for name, value in restored.state_dict().items()))
            self.assertTrue(all(sha256(base / name) == entry["sha256"] for name, entry in metadata["files"].items()))
            with self.assertRaises(FileExistsError):
                prepare_loading_view(base, metadata, derived, view)


if __name__ == "__main__":
    unittest.main()
