# Evidence index

All paths are relative to `reproduction/`. Historical source paths inside records are provenance, not installation requirements.

| Paper reference | Packaged evidence |
|---|---|
| E1 | `docs/self-replicating-organism-plan-20261013.md` |
| E2 | `docs/replicate-first-lineage-plan-20261016.md` |
| E3 | `evidence/publication-support/completion/completion-receipt.json`; `evidence/publication-support/completion/final-report.tar.gz` |
| E4 | `evidence/history/completed-runtime-gates.json` |
| E5 | `evidence/history/lineage-measurements.json` |
| E6 | `evidence/original-native/independent-native-audit.json`; `evidence/original-native/external-native-admission.json` |
| E7 | `evidence/original-lineages/08/independent-lineage-audit.json`; `evidence/original-lineages/09/independent-lineage-audit.json`; `evidence/original-lineages/10/independent-lineage-audit.json` |
| E8 | `evidence/publication-support/training/run.json`; `evidence/publication-support/training/metrics.jsonl`; `evidence/publication-support/training/executed-contract.json`; `evidence/publication-support/training/executed-recipe.json`; `evidence/publication-support/training/executed-input-kit.tar.gz` |
| E9 | `evidence/publication-support/training/corpus-manifest.json` |
| E10 | `evidence/publication-support/export/independent-response-audit.json`; `evidence/publication-support/export/merge/report.json`; `evidence/publication-support/export/export/report.json` |
| E11 | `evidence/publication-support/development-comparisons/comparison-history.tar.gz`; `evidence/publication-support/development-comparisons/final-model-gate-reconstruction.json` |
| E12 | `evidence/original-native/returned/temperature-tasks/temperature-cases.frozen.json` |
| E13 | `evidence/publication-support/recovery/RELEASE-RECOVERY.json`; `evidence/publication-support/recovery/release-recovery.tar.gz` |
| E14 | `evidence/history/unsuccessful-lineage-index.json` |
| E15 | `evidence/history/native-history-index.json` |
| E16 | `src/organism/application.py` |
| E17 | `docs/FIRST_LINEAGE_OPERATIONS.md` |
| E18 | `evidence/holdout/results.json`; `evidence/history/holdout-02` |
| E19 | `evidence/mutations/results.json` |
| E20 | `evidence/history/native-controls-20261017/independent-control-audit.json`; `evidence/history/native-controls-20261017-inputs`; `evidence/history/native-control-allocation-01` |
| E21 | `README.md`; `paper.tex`; `results/paper-summary.json` |
| E22 | `evidence/controls/independent-control-audit.json`; `evidence/control-protocol-20261018/protocol.json`; `evidence/control-allocation-20261018/CLOSED.json`; `evidence/history/lifecycle-correction-20261018/frozen-comparison.json` |

The training input kit is the exact executed snapshot. Its embedded contract, not a later standalone freeze, matches the training run. Full base-model weights, the original GPU environment, and a new fine-tuning execution are not supplied or claimed. Saved model outputs and reference logits support inspection and recalculation of the export numbers.
