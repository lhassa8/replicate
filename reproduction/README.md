# Replicate: reproducibility package

Lars Tray, Georgetown University  
October 18, 2026

This package accompanies the final manuscript in `../paper/replicate.pdf`.
It contains no claim of public posting or replication by an unaffiliated researcher. The paper source, experiment inputs, model files, runtimes, recorded
outcomes, failed attempts, and reconstruction programs are included.

The synthetic diagnostic result is **40/40 trained, 32/40 stock, and 40/40 fixed
policy**. The auditor experiment accepted **24/24 unmodified controls** and
rejected **108/108 prescribed mutations**. These are descriptive results for
the specified cases, not population reliability or general security estimates.

The corrected October 18 native comparison reconstructed both descendant births
and all **15/15 diagnostics for fixed policy and for the trained model**.
Fixed policy started no inference process and had zero model calls across its
diagnostic and reproduction records. The stock seed passed **5/5 diagnostics**,
then put its reproduction action in ordinary text; no executable choice,
reservation, or component transfer followed. Its ten descendant diagnostics
were unexecuted. These are one cohosted lineage attempt per arm, not a
population success-rate estimate or another separate-machine independence trial.

The October 17 comparison is retained separately. Its trained arm completed
both births and 15/15 diagnostics; stock passed five seed diagnostics but did
not emit an executable reproduction action. The fixed adapter stopped before
its first diagnostic because two lifecycle guards required model inference.
The October 18 freeze integrates the previously tested policy-liveness
correction and repeats every arm once. Only two carried source members differ;
weights, prompt templates, tools, cases, resource limits, arm order, and deadlines
remain matched. Fresh release and enrollment identities distinguish the experiments
and change incidental strings in reproduction prompts.
The earlier observations are not pooled with the corrected run.

## Contents and scope

- `paper.tex`: a byte-identical copy of the canonical `../paper/replicate.tex`.
  The final typeset manuscript is `../paper/replicate.pdf`.
- `src/organism`: the application, original auditors, and research programs.
  Experiment directories also contain the source snapshots bound by their
  original protocols.
- `evidence/holdout`: forty authored cases per arm, raw model streams, tool
  events, requests, timings, source and weight identities, and saved scores.
- `evidence/mutations`: six original birth-record subsets and all valid and
  mutated auditor outcomes.
- `evidence/original-lineages`: original files used by the device and birth
  auditors, including the 180 lineage diagnostics and 60 endurance diagnostics.
  Cloud admission receipts remain historical observations; replay does not
  make new cloud observations.
- `evidence/original-native`: the original single-device audit's recorded
  inputs and twenty diagnostic cases. Diagnostic replay adds these to the
  240 lineage and endurance executions, reconstructing all 260 reported
  diagnostic executions. Its cold-start and cloud admission receipts remain
  historical records, not a new admission.
- `evidence/controls`: the corrected native control records, with reconstructed
  diagnostics, birth links, zero-model checks, and observed timings.
- `evidence/control-protocol-20261018`: the new freeze and its source snapshot.
- `evidence/control-allocation-20261018`: input admission, execution permit,
  returned archive identity, resource checks, and final cleanup.
- `evidence/history/native-controls-20261017`: the complete earlier comparison,
  including its fixed-adapter defect and stock action-format failure.
- `evidence/history/native-controls-20261017-inputs`: the earlier signed
  manifests and source archives needed to verify the exact correction.
- `evidence/history`: previous sandbox and bootstrap attempts, original failure
  indices, experiment reservations, and cleanup records.
- `assets`: both exact compact models; carried Linux x86-64 and ARM64 runtimes;
  the Mac inference runtime; frozen capsule components; public signature
  verification keys; and the comparison owner program.
- `evidence/publication-support`: compact completion, training, export,
  development-comparison and recovery records supporting the detailed paper claims.
- `EVIDENCE-INDEX.md`: a package-relative map for all project citations E1-E22.
- `results`: derived tables, per-case CSV, and historical verification receipts.
  Current publication checks are in `../verification/`.
- `MANIFEST.json`: SHA-256 and byte count for every packaged file. The separate
  outer manifest covers the complete publication ZIP, including this manifest.

ARM64 assets being included does not establish native ARM64 qualification.
The package supports inference and recorded-evidence reconstruction, not the
entire historical fine-tuning run. The exact executed training input kit and
selected run/export outputs are included for inspection; the original full base
weights and historical GPU environment are not. Both compact model files are
supplied explicitly. Carried Linux Python is 3.12.14; local replay uses 3.12.13.

## Verify and reconstruct recorded evidence

Use Python 3.12. The original local environment used Python 3.12.13.
`requirements-lock.txt` records its exact installed dependency versions;
`requirements.txt` lists the four directly frozen research dependencies.
Run all commands below from the `Replicate/reproduction` directory. From the
extracted bundle root, enter it with `cd reproduction`. Create an environment
outside the extracted package:

```sh
python3.12 -m venv /tmp/replicate-replay-env
/tmp/replicate-replay-env/bin/python -m pip install -r requirements-lock.txt
export PYTHONPATH="$PWD/src"
export OTEL_SDK_DISABLED=true
export AWS_EC2_METADATA_DISABLED=true
```

Dependency installation may contact a package registry. The following evidence
reconstruction commands do not start a model, contact AWS, or create a descendant:

```sh
/tmp/replicate-replay-env/bin/python -m organism.research.verify_package --package "$PWD"
/tmp/replicate-replay-env/bin/python -m organism.research.replay_evidence holdout \
  --source evidence/holdout --output /tmp/replicate-holdout-replay.json
/tmp/replicate-replay-env/bin/python -m organism.research.replay_evidence mutations \
  --source evidence/mutations --output /tmp/replicate-mutation-replay.json
/tmp/replicate-replay-env/bin/python -m organism.research.replay_original_lineages \
  --evidence evidence --output /tmp/replicate-original-lineage-replay.json
/tmp/replicate-replay-env/bin/python -m organism.research.audit_controls \
  --source evidence/controls --cases assets/smoke-cases.json \
  --output /tmp/replicate-native-control-replay.json
/tmp/replicate-replay-env/bin/python -m organism.research.describe_control_run \
  --source evidence/controls --output /tmp/replicate-native-description-replay.json
/tmp/replicate-replay-env/bin/python -m organism.research.control_outcomes \
  --source evidence/controls --output /tmp/replicate-native-failure-replay.json
/tmp/replicate-replay-env/bin/python -m organism.research.summarize_paper \
  --evidence evidence --output /tmp/replicate-paper-tables
/tmp/replicate-replay-env/bin/python -m organism.research.summarize_control_timings \
  --audit evidence/controls/independent-control-audit.json \
  --description evidence/controls/control-description.json \
  --output /tmp/replicate-native-timing-table
```

The earlier experiment and the exact source correction can also be replayed:

```sh
/tmp/replicate-replay-env/bin/python -m organism.research.audit_controls \
  --source evidence/history/native-controls-20261017 \
  --cases evidence/history/native-controls-20261017/smoke-cases.json \
  --output /tmp/replicate-previous-native-audit.json
/tmp/replicate-replay-env/bin/python -m organism.research.control_outcomes \
  --source evidence/history/native-controls-20261017 \
  --output /tmp/replicate-previous-native-failure.json
/tmp/replicate-replay-env/bin/python -m organism.research.verify_control_correction \
  --previous evidence/history/native-controls-20261017-inputs \
  --current assets --output /tmp/replicate-source-correction.json
```

Compare the reconstructed native JSON with the saved audit, and the newly
derived tables with `results/paper-summary.json`. Expected failures are part
of a matching reconstruction. The original lineage replay verifies per-device,
diagnostic, endurance, and parent-child record consistency; it does not repeat
the original infrastructure-admission process or attest to currently live hosts.

For a source-and-evidence-only extraction, skip `assets/` and use
`verify_package --evidence-only`. The other reconstruction commands still work
when the smoke cases are taken from `evidence/controls/smoke-cases.json`.
Large assets are required for new inference and complete package verification.
The source-correction check only needs the two small code archives and
`assets/assets.json`, in addition to the retained earlier inputs.

## Execute the published diagnostic cases again

On an Apple Silicon Mac with the pinned Python environment and enough memory
for one model, use an output directory that does not already exist:

```sh
/tmp/replicate-replay-env/bin/python -m organism.research.rerun_holdout \
  --package "$PWD" --output /tmp/replicate-new-diagnostic-run
```

The pinned CPU runtime and both weight identities are checked before execution.
The program runs forty fixed-policy cases and eighty model cases in the original
block order, retaining errors and unexecuted cases. This is a new run on published
cases, not a fresh blind holdout. Elapsed times depend on the host and load.
The original run took about forty-three minutes.

## Execute the frozen Linux deployment comparison again

Use a fresh, disposable x86-64 Ubuntu 24.04 guest with systemd, at least 16 GiB
RAM, eight CPU cores, and 96 GiB temporary disk. Extract the full package onto
that guest. The owner program requires root to create three local system users
and bounded services; applications run as unprivileged users. No cloud API or
network download is required after the package is present.

```sh
sudo python3.12 src/organism/research/rerun_controls.py \
  --package "$PWD" --work /opt/replicate-reproduction
```

The wrapper verifies frozen assets, uses the carried Python and dependencies,
executes arms sequentially, applies a three-hour owner timeout, and stops its
application services on exit. The disposable guest remains the experiment
owner's responsibility. This is the cohosted control design, not the original
three-machine independence trial. This entry point runs the corrected October 18
freeze. Its class names retain
the word `Candidate` to preserve the exact tested source identity. New outcomes
are determined by execution; saved successes are not guarantees for a new run.
Audit the new `results` directory with `audit_controls`.

The original separate-machine deployment procedure is retained in
`docs/FIRST_LINEAGE_OPERATIONS.md`. Reconstruct the exact original capsule
without substituting the supplementary comparison's application archive:

```sh
/tmp/replicate-replay-env/bin/python -m organism.research.reconstruct_capsule \
  --package "$PWD" --output /tmp/replicate-original-capsule
```

This verifies all eleven components and the original signature before producing
the 2.86 GB payload directory. It does not provision receivers or start a model.
`--verify-inputs-only` verifies the same inputs without creating another copy.
Fresh trials require newly generated
enrollment and signing identities, compatible owner receivers, and separately
bounded infrastructure. Historical private credentials are not provided.

## Numerical export figures

After the same dependency setup, run:

```sh
/tmp/replicate-replay-env/bin/python replay_export_numerics.py
```

This reconstructs the reported merge/probability figures from the twelve saved
responses and reference logits, using the historical numerical auditor. It does
not rerun inference, training, full export admission, or tensor-layout validation.
The supporting records are in `evidence/publication-support/export/`.

## Publication and license status

The outer bundle includes the manuscript, arXiv source upload and posting guide.
It has not been publicly deposited or submitted on the author's behalf.
Author-selected licenses and their exact scope are in `../LICENSES.md`. Bundled third-party assets retain their own licenses,
notices and the additional source support in `../third-party/`.

Credential-bearing enrollment stores, private signing keys, presigned transfer
URLs and chat history are excluded. Historical paths and public resource
identifiers remain where needed for evidence provenance. Preserved status flags
and original verification receipts describe their dated experiments; current
publication verification is recorded separately in `../verification/`.
