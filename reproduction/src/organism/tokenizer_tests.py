"""Run separately with organism/.venv-data/bin/python -m organism.tokenizer_tests."""

import os
import json
from pathlib import Path
import shutil
import tempfile
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoTokenizer
from .tokenize_dataset import completion_tokens
from .tokenize_dataset import sha256
from .check_runtime_templates import check


class NativeTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "evidence/tokenizer-source-20261013-01"
        cls.tokenizer = AutoTokenizer.from_pretrained(path.resolve(), local_files_only=True)
        cls.prompt = [{"role": "user", "content": "Inspect device example."}]
        cls.action = {
            "role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "id": "test_0", "function": {
                "name": "inspect_device", "arguments": {"device_id": "example"},
            }}],
        }

    def test_only_complete_assistant_action_is_supervised(self):
        full, labels, prefix, text = completion_tokens(self.tokenizer, self.prompt, self.action, [])
        self.assertTrue(all(label == -100 for label in labels[:prefix]))
        self.assertEqual(full[prefix:], labels[prefix:])
        self.assertIn("<parameter=device_id>\nexample\n</parameter>", text)
        self.assertIn("</tool_call>", text)
        self.assertIn(self.tokenizer.eos_token_id, labels[prefix:])

    def test_overlong_context_is_rejected_not_truncated(self):
        long_prompt = [{"role": "user", "content": "device " * 5000}]
        with self.assertRaisesRegex(ValueError, "exceeds contract"):
            completion_tokens(self.tokenizer, long_prompt, self.action, [])

    def test_non_tool_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "tool call"):
            completion_tokens(
                self.tokenizer, self.prompt,
                {"role": "assistant", "content": "Done."}, [],
            )

    def test_overlong_action_rejected_not_truncated(self):
        action = {
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "inspect_device", "arguments": {"device_id": "value " * 250}}}],
        }
        with self.assertRaisesRegex(ValueError, "exceeds contract"):
            completion_tokens(self.tokenizer, self.prompt, action, [])

    def test_final_acknowledgement_is_complete_and_prompt_is_masked(self):
        final = {"role": "assistant", "content": "Assessment recorded."}
        full, labels, prefix, text = completion_tokens(self.tokenizer, self.prompt, final, [], target_kind="final")
        self.assertTrue(all(label == -100 for label in labels[:prefix]))
        self.assertEqual(full[prefix:], labels[prefix:])
        self.assertIn("Assessment recorded.", text)
        self.assertIn(self.tokenizer.eos_token_id, labels[prefix:])
        self.assertNotIn("<tool_call>", text)

    def test_final_cannot_hide_a_tool_call_or_omit_acknowledgement(self):
        for final in (self.action, {"role": "assistant", "content": ""},
                      {"role": "assistant", "content": "<tool_call>invented</tool_call>"}):
            with self.assertRaisesRegex(ValueError, "acknowledgement"):
                completion_tokens(self.tokenizer, self.prompt, final, [], target_kind="final")

    def test_nonassistant_and_unknown_target_kinds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "target kind"):
            completion_tokens(self.tokenizer, self.prompt, self.prompt[0], [], target_kind="final")
        with self.assertRaisesRegex(ValueError, "target kind"):
            completion_tokens(self.tokenizer, self.prompt, self.action, [], target_kind="anything")

    def test_json_scalar_codec_changes_target_and_keeps_every_prompt_token(self):
        action = {
            "role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "submit", "arguments": {
                    "value": None, "valid": False, "note": "None"}}}]}
        old, _, old_prefix, old_text = completion_tokens(self.tokenizer, self.prompt, action, [])
        full, labels, prefix, text = completion_tokens(
            self.tokenizer, self.prompt, action, [], target_codec="granite-json-scalar-targets-v1")
        self.assertEqual(prefix, old_prefix)
        self.assertEqual(full[:prefix], old[:prefix])
        self.assertEqual(labels[:prefix], [-100] * prefix)
        self.assertEqual(labels[prefix:], full[prefix:])
        self.assertIn("<parameter=value>\nNone\n</parameter>", old_text)
        self.assertIn("<parameter=value>\nnull\n</parameter>", text)
        self.assertIn("<parameter=valid>\nfalse\n</parameter>", text)
        self.assertIn("<parameter=note>\nNone\n</parameter>", text)
        self.assertIn(self.tokenizer.eos_token_id, full[prefix:])
        self.assertEqual(self.tokenizer.encode(self.tokenizer.decode(old[:prefix]) + text,
                                              add_special_tokens=False), full)

    def test_unknown_target_codec_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "target codec"):
            completion_tokens(self.tokenizer, self.prompt, self.action, [], target_codec="invented")


class RuntimeTokenArtifactTests(unittest.TestCase):
    """Independently audit actual serving prefixes and reject rehashed bad rows."""

    root = Path(__file__).parent
    token_directory = root / "data/runtime-development-v2-tokens-v3"
    captures = root / "evidence/runtime-template-preflight-20261014-02"
    tokenizer_path = root / "evidence/tokenizer-source-20261013-01"
    source = root / "data/runtime-development-v2"

    def test_serving_prefixes_and_native_targets_verify_despite_template_difference(self):
        report = check(self.captures, self.token_directory, self.tokenizer_path, self.source)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["requests"]), 1200)
        self.assertFalse(report["unmodified_native_prompt_parity_verified"])
        self.assertEqual(sum(not row["unmodified_native_text_equal"] for row in report["requests"]), 12)

    def mutate_and_check(self, mutation):
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary) / "tokens"
            shutil.copytree(self.token_directory, copy)
            path = copy / "train.tokens.jsonl"
            lines = path.read_text().splitlines()
            row = json.loads(lines[0])
            mutation(row)
            lines[0] = json.dumps(row, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n")
            report_path = copy / "report.json"
            report = json.loads(report_path.read_bytes())
            report["splits"]["train"]["sha256"] = sha256(path)
            report_path.write_text(json.dumps(report))
            result = check(self.captures, copy, self.tokenizer_path, self.source)
            self.assertFalse(result["passed"])
            self.assertFalse(result["requests"][0]["entire_row_and_mask_verified"])

    def test_prompt_loss_leak_rejected_even_after_updating_the_file_hash(self):
        self.mutate_and_check(lambda row: row["labels"].__setitem__(0, row["input_ids"][0]))

    def test_self_consistent_wrong_target_rejected_against_original_reference(self):
        def mutation(row):
            index = row["prompt_tokens"]
            row["input_ids"][index] += 1
            row["labels"][index] = row["input_ids"][index]
        self.mutate_and_check(mutation)


class EquipmentTokenArtifactTests(unittest.TestCase):
    root = Path(__file__).parent
    tokens = root / "data/equipment-runtime-development-v1-tokens"
    captures = root / "evidence/equipment-runtime-template-preflight-20261014-01"
    tokenizer_path = root / "evidence/tokenizer-source-20261013-01"
    source = root / "data/equipment-runtime-development-v1"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.tokenizer_path.resolve(), local_files_only=True)
        cls.contracts = json.loads((cls.source / "contracts.json").read_bytes())
        cls.records = [json.loads(line) for split in ("train", "validation")
                       for line in (cls.source / (split + ".jsonl")).read_text().splitlines()]
        cls.rows = [json.loads(line) for split in ("train", "validation")
                    for line in (cls.tokens / (split + ".tokens.jsonl")).read_text().splitlines()]

    def test_all_equipment_corpus_targets_match_actual_serving_and_native_templates(self):
        report = check(self.captures, self.tokens, self.tokenizer_path, self.source)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["requests"]), 1440)
        self.assertTrue(report["unmodified_native_prompt_parity_verified"])
        self.assertEqual(sum(row["target_kind"] == "final" for row in self.rows), 320)

    def test_bad_actions_are_masked_history_and_corrected_actions_remain_targets(self):
        errors = [record for record in self.records if False in record.get("supervised_assistants", [])]
        self.assertEqual(len(errors), 8)
        for record in errors:
            with self.subTest(case=record["case_id"]):
                rows = {row["decision_index"]: row for row in self.rows if row["case_id"] == record["case_id"]}
                self.assertEqual(set(rows), {0, 2, 3, 4, 5})
                assistant_indices = [index for index, message in enumerate(record["messages"])
                                     if message["role"] == "assistant"]
                bad_index, correction_index = assistant_indices[1:3]
                _, _, _, bad_text = completion_tokens(
                    self.tokenizer, record["messages"][:bad_index], record["messages"][bad_index],
                    self.contracts["equipment"]["tools"])
                row = rows[2]
                prefix = row["prompt_tokens"]
                prefix_text = self.tokenizer.decode(row["input_ids"][:prefix])
                self.assertIn(bad_text, prefix_text)
                self.assertIn("ValidationError", prefix_text)
                self.assertEqual(row["labels"][:prefix], [-100] * prefix)
                ids, _, boundary, _ = completion_tokens(
                    self.tokenizer, record["messages"][:correction_index], record["messages"][correction_index],
                    self.contracts["equipment"]["tools"])
                self.assertEqual(row["labels"][prefix:], ids[boundary:])
                self.assertEqual(rows[5]["target_kind"], "final")
                self.assertIn(self.tokenizer.eos_token_id, rows[5]["labels"])

    def test_rehashed_source_cannot_promote_injected_error_into_a_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, capture, tokens = (root / name for name in ("source", "capture", "tokens"))
            for destination, original, names in (
                (source, self.source, ("manifest.json", "train.jsonl", "validation.jsonl", "contracts.json", "release.json")),
                (capture, self.captures, ("report.json", "server-prompts.jsonl")),
                (tokens, self.tokens, ("report.json", "train.tokens.jsonl", "validation.tokens.jsonl")),
            ):
                destination.mkdir()
                for name in names:
                    shutil.copy2(original / name, destination / name)
            path = source / "train.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            record = next(item for item in records if False in item.get("supervised_assistants", []))
            record["supervised_assistants"][1] = True
            path.write_text("\n".join(json.dumps(item, separators=(",", ":")) for item in records) + "\n")
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["files"]["train.jsonl"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
            manifest_path.write_text(json.dumps(manifest))
            capture_path = capture / "report.json"
            report = json.loads(capture_path.read_bytes())
            report["source_manifest_sha256"] = sha256(manifest_path)
            capture_path.write_text(json.dumps(report))
            token_path = tokens / "report.json"
            report = json.loads(token_path.read_bytes())
            report.update(source_manifest_sha256=sha256(manifest_path),
                          serving_capture_report_sha256=sha256(capture_path))
            token_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "bad action entered targets"):
                check(capture, tokens, self.tokenizer_path, source)

    def test_rehashed_tokens_cannot_omit_correction_or_final_acknowledgement(self):
        record = next(item for item in self.records if False in item.get("supervised_assistants", []))
        for ordinal in (2, 5):
            with self.subTest(ordinal=ordinal), tempfile.TemporaryDirectory() as temporary:
                tokens = Path(temporary) / "tokens"
                shutil.copytree(self.tokens, tokens)
                path = tokens / "train.tokens.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows = [row for row in rows if (row["case_id"], row["decision_index"]) != (record["case_id"], ordinal)]
                path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")
                report_path = tokens / "report.json"
                report = json.loads(report_path.read_bytes())
                report["splits"]["train"]["sha256"] = sha256(path)
                report["splits"]["train"]["decisions"] -= 1
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "Extra or duplicate server prompt"):
                    check(self.captures, tokens, self.tokenizer_path, self.source)


if __name__ == "__main__":
    unittest.main()
