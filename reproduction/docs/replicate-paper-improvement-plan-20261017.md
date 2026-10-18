# Replicate paper improvement experiment

Authorized October 17, 2026: the user accepted the five-part paper review and
requested execution. This is a research supplement to the completed reduced
milestone. It does not reopen the original Q1–Q10 qualification.

## Questions and completion criteria

1. What do trained, stock, and deterministic policies contribute under matched
   deployment and diagnostic conditions? Preserve the original release and
   report all control outcomes, including failures and inference overhead.
2. Does the frozen diagnostic system work on unseen synthetic descriptor
   families? Freeze source, prompts, weights, and the experiment before running
   40 cases from four new layout families. Do not tune on these results.
3. Which invalid replication claims do the existing auditors reject? Evaluate
   frozen auditors against valid records and explicit mutations, including
   semantically inconsistent records with recomputed outer hashes. Report
   false acceptance and false rejection without general security claims.
4. How does the method differ from existing replication evaluations? Verify
   primary literature and add a concrete boundary comparison.
5. Can someone reproduce the evidence outside this workspace? Build a
   sanitized package with frozen code, inputs, results, auditors, dependencies,
   and instructions; test it in a separate clean directory. A replay performed
   by this assistant is not unaffiliated replication. Prepare public release
   material without claiming that private artifacts are publicly available.

## Experiment discipline

- Keep the completed release and all earlier evidence unchanged.
- Keep new material under `organism/research/` and a new evidence directory.
- Use the existing model pair and tool implementation; no training.
- The deterministic arm must have zero model calls. Its common deployment
  and useful-task outcomes are separate from the model-specific predicates.
- Preserve every attempted run. Infrastructure fixes do not justify tuning
  the frozen models, prompts, cases, or scoring.
- Original 20 cases remain development data. The new 40 cases are a one-shot,
  author-designed synthetic holdout, not an external benchmark or a new task
  domain. Four families are not 40 independent deployment environments.
- Keep native deployment, local host controls, fixture tests, trace replay,
  and external replication explicitly distinct in reporting.
- Cloud work retains the standing combined reservation ceiling below $400.
  Previous reservations are $367, leaving at most $32 for this supplement.
  No launch without a concrete resource/time estimate, fixed independent
  expiry, exact input identities, and cleanup procedure. No GPU is planned.
- No public submission or external message is included in execution.

## Paper deliverable

Revise the open `output/pdf/replicate-whitepaper-20261017.tex` in place.
Organize the main text around the research questions and measured answers;
retain original-versus-reduced scope and move operational chronology to
appendices where appropriate. Include exact controls, one worked trace,
network shaping, limitations, and accessible-artifact status. Compile with
the document editor's built-in compiler. Do not create a replacement PDF.

## Execution log

- Initial inspection: original matched compact models and completed native
  capsule are present locally. Live AWS inspection found no active EC2
  instances in the four enabled commercial US regions. Unrelated stopped
  instances were left untouched.
- Existing auditor negative tests cover actual TLS and byte transfer with
  explicitly simulated model/process outcomes. Supplementary native-record
  mutation results must remain distinguishable from those fixtures.
- Native-record experiment completed: 24/24 positive checks accepted and
  108/108 invalid variants rejected across six original births. Protocol,
  accessed evidence subsets, source identities, and every result are in
  `organism/evidence/paper-auditor-mutations-20261017-01`.
- The diagnostic freeze binds the final native application, preserving three
  documented changes made after the earlier development comparison. The new
  equipment prompt and equipment tools match the final comparison. A sandbox
  attempt started no model processes; the complete unchanged experiment is
  completed under the required local permissions in
  `organism/evidence/paper-heldout-20261017-03`: trained 40/40, stock 32/40,
  fixed policy 40/40. The eight stock failures are the low and high conditions
  in every family. There was no post-outcome model or prompt change.
- A new Python environment installed the pinned dependencies from a package
  registry. With copied source and evidence outside the project directory,
  the holdout scores, all 132 auditor outcomes, and the original three lineages'
  180 diagnostic plus 60 endurance results reconstructed exactly.
- First native allocation: startup stopped before application execution because
  the narrow storage role received AccessDenied for the missing permit. The
  host, disk, network, IAM roles, schedule, and stack were removed and directly
  checked. A corrected bootstrap and explicit public-directory traversal were
  frozen for the second, final allocation. Application policies, models,
  prompts, cases, and scoring remained identical.
- Second native allocation: the fixed-policy application exposed a complete-loop
  integration defect and exited before diagnostics. Its correctly false
  inference-readiness flag was rejected by the inherited model supervisor.
  The stock arm completed all five seed diagnostics, then reached its unchanged
  20-minute child deadline. The failed decision trace correctly inspected the
  eligible receiver, but expressed its establishment action as text rather
  than an executable tool call. No reservation or transfer began.
  The trained arm completed its lineage and all 15 diagnostics. Both complete
  eleven-component transfers and seed removal before grandchild availability
  reconstructed under the original model-choice and link checks.
  A separate policy-liveness correction passes four local lifecycle checks,
  including real TLS transfer with explicit fixture readiness; it is not part of the native
  experiment and has not been promoted to a validated baseline.
- Native output preservation verified the returned archive's SHA-256 and byte
  count before deleting temporary storage. Both worker allocations, disks,
  network resources, roles, schedules, and the shared artifact bucket were
  directly checked as terminated or absent. The supplement's observed
  instance-lifetime upper bound is 1.094902 hours across two allocations;
  this is not a billing invoice. The $20 supplement reservation and $387
  combined reservation remained unchanged.
- The new reproduction wrapper was checked against all eleven original signed
  capsule components. Original diagnostic replay now reconstructs all 260
  reported executions: 180 lineage, 60 endurance, and 20 single-device cases.

## Interpretation after execution

The five improvement areas have produced a narrower research contribution,
new executed evidence, retained negative results, and a portable reproduction
candidate. The trained checkpoint passed the forty new synthetic diagnostic
cases and the cohosted lineage. It does not outperform the fixed diagnostic
solver. The fixed native adapter's failure means the full matched deployment
question remains unanswered; the local correction candidate needs a new signed
freeze and a complete three-arm native rerun. The stock native outcome is one
observed action-format failure, not a reliability estimate.

External task design, unaffiliated replication, a public artifact destination,
and the author's project-code license remain research or publication follow-up.
The supplement does not qualify the original heterogeneous population plan.

## Bounded deployment control design

Use one c7a.2xlarge native x86-64 Linux guest, with separate unprivileged
application accounts, separate signed enrollment and receiving stores, and
an 8 GiB memory ceiling plus four-core CPU quota per application. At most
two model applications run simultaneously; the seed application and receiver
must be stopped before the grandchild becomes available.

Execute fixed, stock, and trained policies sequentially. Each receives the
same application code and uses the same authenticated transport and preparation
code. The stock arm changes only the compact model and its truthful training
declaration. The fixed arm carries the same complete trained payload but never
starts inference; it has an explicit separate policy profile and zero model
streams. All arms use the same five original development smoke conditions
per individual. Those checks are separate from the 40 unseen cases.

This is a cohosted controlled comparison of deployment and diagnostic behavior.
It does not replace the original three separate-machine independence trials,
establish physical-device isolation, or add a fourth passing original lineage.
Retain arm order as a limitation; do not claim a reliability estimate from
one lineage per arm.

Reserve $20 for up to two sequential bounded CPU allocations, at most four
hours each, one active host, 96 GiB encrypted temporary storage, at most 6 GiB
of reusable input uploads and 256 MiB of output per allocation. The second
allocation is allowed only for an evidenced infrastructure correction after
the first host is gone. No policy, model, diagnostic case, or scoring changes
are permitted in response to outcome failures. Maximum combined reservations
become $387, below the standing $400 ceiling. The prior milestone's 27.818
CPU-hour bound remains closed; this separate supplement has its own eight-hour
maximum. Actual c7a.2xlarge Linux On-Demand pricing was read from AWS Pricing:
$0.41056 per hour, with an available Canonical Ubuntu 24.04 x86-64 image.
Retain direct resource readback, fixed guest and independent AWS expiry,
instance-bound permit, output checksums, and complete cleanup evidence.
