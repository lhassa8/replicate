# Replicate: first useful independent lineage

Approved direction: October 16, 2026, in the recovery conversation.
The user agreed to a smaller demonstration and requested a plan, a Codex goal,
and execution. This is a separate development milestone. The original plan at
`docs/self-replicating-organism-plan-20261013.md`, its sealed evaluation inputs,
and Q1–Q10 remain preserved for later qualification.

## Outcome

Demonstrate one reliable, self-contained individual on an 8 GiB x86-64 Linux
device, then repeat an independent seed → child → grandchild lineage three
times. Use one read-only temperature diagnostic interface and the existing
approximately 3B trained model family. Finish with a two-hour endurance test
on the same release used for the passing lineages.

A child must discover an enrolled, newly available device, supply its complete
package, and establish a useful grandchild after the seed is disconnected.
Each running individual performs actual local model inference. The observer
may introduce devices, disconnect the seed, and collect evidence; it cannot
choose reproductive actions, supply missing payloads, or launch descendants.

## Scope

- One native Linux architecture: x86-64. Preserve the already built ARM64
  artifacts, but defer cross-architecture testing.
- Existing IBM Granite 4.2-3B family, with 3,659,737,600 base parameters.
  Evaluate the completed v5 refinement before another training experiment.
- One equipment interface: temperature diagnostics, including normal, fault,
  stale, invalid, and missing observations.
- At most three simultaneously enrolled device slots for this milestone.
  Keep authenticated enrollment, distinct device identities, ownership,
  fixed resource allowances, package verification, stop, and retirement.
- Keep the existing 8 GiB device limits and original 20-minute birth budget.
  Measure actual memory, storage, latency, and model/tool activity.
- No hosted inference, shared model server, runtime package registry, or
  central deployment controller. Owner provisioning of the documented minimal
  receiving substrate is an explicit prerequisite.
- Defer 4 GiB, broad equipment adaptation, four-generation chains, cooperative
  recovery experiments, inherited procedures, AgentCore, and the 72-hour run.
  None of those deferred requirements will be described as passed.

## Execution sequence and gates

### 1. Recover the interrupted work

Inspect live AWS state before acting. Preserve the completed v5 CPU export and
the fourth native run, verify complete checksums and exact retained versions,
and finish their temporary-resource cleanup. Reconcile their ledgers with live
state. Preserve original failures and the service-group correction.

Independently reconstruct the v5 export using the existing numerical limits,
raw responses, calibration inputs, and actual tensor precision. A worker's
successful exit is insufficient. A failed export cannot enter the task runner.

### 2. Establish model behavior on one task

Run the original matched ten-case comparison with the admitted v5 recipe,
stock model, and fixed policy. Preserve the v4 result and report every case,
including the inspection cases that are outside this smaller milestone.

Before running a new temperature evaluation, freeze twenty concrete cases:
four variants in each of the five categories above. Use development-only
fixtures with exact expected tool evidence and diagnoses. Do not read or reuse
sealed qualification cases. Keep these new evaluation cases out of training.
Compare all arms using identical tools, prompts, budgets, and inputs.

Passing requires all twenty temperature cases to complete correctly within
their declared budgets, including `unknown` with a null value for missing,
invalid, or stale evidence. Numerical calculations and validity checks may
use explicit deterministic tools; any such interface change is versioned,
tested, and applied equally to the compared arms.

If v5 fails, diagnose the actual traces before proposing another training run.
Keep work focused on this task and its software/model boundary. Do not mask
model failures with scripted model responses or replace the trained candidate
with a larger or hosted model. A further training experiment needs a concrete
failure hypothesis, independent training examples, and a recorded budget.

### 3. Make one native individual work

Verify the saved service-group correction, rebuild and verify the complete
release, and run it on one bounded x86-64 8 GiB host. Obtain real local inference
and correct task outcomes through the installed application.

Require three successful cold starts with complete local artifacts and no
runtime external dependency. Exercise explicit stop and persistent retirement.
Do not launch a population test while this gate is failing.

### 4. Demonstrate independent lineages

Run three fresh trials with the same release. Start a healthy seed, introduce
an eligible child, and verify that the seed establishes a useful child.
Disconnect the seed before introducing the grandchild. Verify that the child
initiates and completes the grandchild's establishment with its own inference
and retained package.

Both descendants must pass the frozen temperature task checks. Preserve every
attempt, model call, tool result, payload identity, ownership transition,
timeout, and observer action. All three trials must pass; no cherry-picking
successful attempts across releases.

### 5. Demonstrate short sustained operation and close the milestone

Run the final child and grandchild for two continuous hours after seed
disconnection. Schedule the five temperature categories repeatedly with
declared intervals and retain outcomes plus resource samples. Require no
incorrect diagnosis, unexplained restart, missing model evidence, exceeded
resource allowance, or dependence on the seed.

Produce an independently reconstructable result report, release identity,
measured resource summary, and concise operating/stop instructions. Verify
temporary cloud cleanup. Mark this milestone complete only when all five
stages pass on the final release. This is not full Q1–Q10 qualification and
does not by itself establish model or fine-tuning superiority over fixed rules.

## Budget and execution discipline

The standing user authorization remains below $400 for combined project
spending and outstanding reservations. The recovered ledger starts with $215
in conservative reservations; retain those until reconciled.

Reserve at most another $60 for this milestone's native compute, ancillary
resources, and transfers, bringing planned reservations to $275. Permit at
most thirty additional CPU instance-hours, at most three active devices,
and fixed per-instance leases of at most four hours. Record exact allocations,
prices, deadlines, and independent expiry before publishing execution permits.
Never extend a running lease. Verify cleanup after each allocation.

The existing retained-evidence allowance is 50 GB with its existing storage
reservation. Check actual versioned bytes before adding archives. Reuse verified
artifacts and tools; avoid new infrastructure and additional GPU runs until
the completed refinement has been evaluated. Any later allocation must remain
within the combined project ceiling and have its own concrete bounds.

Changes stay under `organism/` and the relevant project documents. Preserve
other work in this shared workspace. Maintain one milestone status record
with evidence pointers and the next executable step.

## Execution amendment: October 16, 2026

The v5 temperature comparison finished at 4/20 trained, 19/20 stock and 20/20
fixed-policy successes. Its complete results remain in
`organism/evidence/first-lineage-temperature-evaluation-20261016-01`.
The trace diagnosis and independently constructed training targets are recorded
in `organism/evidence/first-lineage-training-preparation-20261016-01`.
The revised recipe is
`organism/training-recipe.first-lineage-v6-02.draft.json`; the twenty evaluation
cases and original acceptance requirements remain unchanged.

Seven bounded EC2 capacity requests allocated no GPU instances. The first
managed GPU allocation failed before training; a bounded CPU diagnostic
established the isolation correction. A separately frozen second managed GPU
allocation completed 90 training steps and passed training admission. Its
evidence and temporary-resource cleanup are recorded in
`organism/evidence/first-lineage-managed-training-20261016-02`.
Completion of training does not establish model quality.

Conservative combined reservations are now $367: the original $275, $35 for
each of the two managed GPU allocations, $2 for the isolation diagnostic, and
$20 for the CPU export. The original $60 native reservation is already included
in that total. Retain these reservations until reconciled; do not count the
unallocated EC2 capacity requests as additional GPU allocations.

The current CPU export has one four-hour lease and a fixed expiry. Its reserved
four CPU instance-hours and the diagnostic's reserved half hour count against
the existing thirty-hour ceiling. Reconcile actual closed lifetimes before
reserving subsequent native trials. No third GPU allocation is included.
The combined ceiling remains strictly below $400, the evidence allowance remains
50 GB, and all original native, lineage, and endurance gates still apply.

### Completed export and runtime instruction experiment

The v6 export passed the unchanged numerical checks and all temporary export
resources were verified removed. Its conservatively measured closed lifetime
was 0.875495 CPU instance-hours.

The complete v6 temperature comparison is preserved in
`organism/evidence/first-lineage-temperature-evaluation-20261016-02`.
Its original runtime audit counted 7/20 trained and 18/20 stock successes.
Independent review found that Mac sleep could age a nominal fixture or exceed
the case's wall-clock budget. The corrected milestone audit counts 6/20 trained,
17/20 stock and 20/20 fixed-policy successes. Both audits remain preserved.
The correction enforces the original frozen conditions and budgets.

Most trained failures repeat shortened timestamp or validity selectors.
A prompt-only experiment explicitly instructs both models to copy every
published selector exactly and correct rejected arguments before retrying.
It changes no weights, training data, tool implementation, schema, cases,
call allowance or acceptance threshold. Preparation and the exact source
change are in `organism/evidence/first-lineage-selector-instructions-20261016-01`.
Fifty focused checks passed. The complete matched twenty-case comparison is
`organism/evidence/first-lineage-temperature-evaluation-20261016-03`, with idle
sleep prevention limited to that run. It must pass before native allocation.

The selector-instruction comparison completed at 19/20 trained, 18/20 stock
and 20/20 fixed-policy successes. Every frozen condition, wall-clock budget
and trace audit passed. The trained model stopped repeating invalid selectors;
its one remaining failure rounded a returned 37.25 C measurement to 37 C.

A second prompt-only experiment explicitly forbids rounding or truncating the
canonical value and requires preservation of every decimal place. Its exact
change and fifty passing focused checks are in
`organism/evidence/first-lineage-value-instructions-20261017-01`.
The full matched comparison is
`organism/evidence/first-lineage-temperature-evaluation-20261016-04`.
The weights, cases, tools, budgets and numerical tolerance remain unchanged.
Prepared selector-feedback code remains isolated and unapplied because it
does not address this remaining failure.

The decimal-copy comparison also finished at 19/20 trained, with 19/20 stock
and 20/20 fixed-policy successes. All trace, frozen-condition and timing
checks passed. The sole trained failure again rounded 37.25 C to 37 C.

A bounded software-boundary experiment adds canonical submission-value
consistency validation. A mismatch is rejected without recording a diagnosis;
the model receives the previously issued canonical value and must resubmit.
The tool does not substitute values or choose diagnosis status. Independent
auditing reconstructs this feedback. The six-call allowance, time budgets,
weights, frozen cases and semantic tolerance remain unchanged. Fifty-four
focused checks passed. Exact before/after sources and application evidence
are in `organism/evidence/first-lineage-value-validation-preparation-20261017-01`.
The full matched comparison is
`organism/evidence/first-lineage-temperature-evaluation-20261016-05`.
Native allocation remains gated on all twenty trained cases passing.

Comparison 05 completed and passed its independent audit on October 17, 2026:
trained 20/20, stock 19/20 and fixed policy 20/20. Every frozen condition,
wall-clock allowance and trace audit passed. The trained model corrected its
37 C submission to 37.25 C after a rejected value, using the original six-call
allowance. The control model corrected the 0 C value but still selected the
wrong diagnosis; that case remained a failure. This result admits the exact
current application and exported weights to the single-device native gate.
It does not establish native execution or general model superiority.

### Native readiness diagnosis and local replay

The first 8 GiB x86-64 allocation started the application and its own model,
but failed its first public readiness diagnostic. The model duplicated part
of the requested device identifier, then submitted the returned error string
as measurement evidence. The checker correctly rejected the result after
128.399 seconds, within its original 240-second deadline. No native cold
start or twenty-case task pass is claimed. Complete traces, exact launch
controls and independent failure reconstruction are preserved in
`organism/evidence/first-lineage-native-execution-20261016-01`.
All temporary compute, storage, network, identity and expiry resources were
verified removed at 01:44:13 UTC on October 17.

A frozen local prompt-only experiment is recorded in
`organism/evidence/first-lineage-readiness-identifier-preparation-20261017-01`.
First replay the exact failed challenge with the original prompt on both
stock and trained models. After that worker exits, apply only the proposed
generic readiness prompt and run the failed challenge plus five predeclared
fresh challenges on both models. Retain every result. All six trained cases,
the literal-trace audits and clean process shutdown must pass. The checker,
four-call allowance, time budgets, long identifiers and model weights remain
unchanged. This local experiment does not admit another native allocation:
the full twenty-case comparison must also be repeated with the final source
inventory before building the next exact release.

The original-prompt replay failed on both models. The identifier-focused
candidate completed all six challenges at 3/6 for both models; all trace and
shutdown audits passed. Its trained failures copied the observations and
evidence correctly but selected `normal` for values below the inspected
range. The native gate remains closed.

The next prompt-only experiment, in
`organism/evidence/first-lineage-readiness-range-preparation-20261017-01`,
adds the explicit inclusive-range rules already used in the successful
equipment-task prompt. The same six challenges, both models, original checker,
weights and allowances are retained. The completed preceding experiment is
independently reconstructed before this follow-up runs.

The range-rule experiment finished at 5/6 trained and 1/6 stock. The remaining
trained case altered the device identifier, recovered, and submitted the
correct low diagnosis on its fourth call. The unchanged checker rejected the
rejected action and incomplete completion trace. A separate negative readback
reconstructs that failure while preserving the common auditor's inability to
equate a correct submission with a failed readiness check.

The next experiment binds `device_id` in each generic diagnostic tool schema
to the exact identifier selected by the request. The identifier remains full
length, literal arguments remain recorded, and a mismatch is rejected without
substitution. Prompt, checker, model, six challenges and budgets are unchanged.
Preparation is in
`organism/evidence/first-lineage-readiness-device-scope-preparation-20261017-01`.
One hundred focused checks passed. All six trained challenges must pass before
the complete temperature comparison and another native allocation.

The device-scoped experiment passed all six trained challenges, with 1/6 for
stock. All literal-trace, full-identifier schema, checker, timing and shutdown
checks passed. The original failed challenge passed without a rejected action.
The complete temperature recheck is
`organism/evidence/first-lineage-temperature-evaluation-20261016-06`.
Its current source-bound result must pass before the fresh native builder in
`organism/evidence/first-lineage-native-capacity-20261017-02` can prepare the
next release. The first failed native allocation remains closed and preserved.

### Optional four-core eight-GiB device

The next candidate selects `c7a.xlarge` explicitly in its native kit. Current
AWS readbacks show the same 8 GiB and four vCPUs, provided by four cores with
one thread per core. The preceding C7i device had two cores with two threads
per core. The purpose is to test whether this gives enough timing margin for
the unchanged diagnostic and lineage budgets; its actual model speed remains
unmeasured. The existing profile defaults and historical templates are
preserved. Eighty-four focused checks passed, including strict verification
of the selected template, described CPU topology, actual instance CPU options
and unchanged memory. This choice adds no reservation beyond the existing
$60 native allowance within the $367 conservative project reservation.
The native builder requires the current six preparation source identities
and the complete final model comparison before producing a candidate.

The final-source comparison 06 completed with trained 20/20, stock 19/20
and fixed policy 20/20. All complete-trace, frozen-condition, timing and
shutdown checks passed; the worker exited successfully. This admits the
current readiness-verified application and unchanged v6 weights to the next
single-device native test. Native execution and reproduction remain unproven.

### Native owner-observer correction

The second single-device run passed its model readiness diagnostic in 61.72
seconds. Independent replay confirmed the literal model/tool trace and correct
diagnosis. The owner observer then failed storage admission because its mount
namespace assigned different mount IDs from those persisted by the receiving
service. The underlying volumes, filesystem identities and paths matched.
No accepted cold start or twenty-case native pass is claimed. The original
failure and both mount views are retained in
`organism/evidence/first-lineage-native-execution-20261017-02`.
All its temporary resources were verified removed at 03:43:31 UTC.

The correction joins only the verified receiving service's mount namespace
before the separate owner observer reads storage. It checks the receiver's
PID and process start identity before and after the transition, verifies the
namespace descriptors, preserves all other observed namespaces, and rejects
an uncertain transition. The persisted storage-admission rules are unchanged.
The independent audits bind the observer's receipt to the actual receiving
service and storage observation. Preparation and 114 passing focused checks
are recorded in
`organism/evidence/first-lineage-owner-mount-preparation-20261017-01`.

The third single-device candidate reuses the exact trained capsule and release
`55e2e098f36f0b0ae3767744fadcee34f7abe01920963c5c9d57eeb3c9c058c2`.
Only its owner kit changes. Its complete comparison and six readiness results
were reconstructed again before preparation. Its fixed expiry is 07:43:22 UTC
on October 17, with the existing memory, task, birth and resource limits.
The native gate still requires three actual cold starts and all twenty frozen
task cases; the later lineage and endurance gates remain pending.
### Native final-collector correction

The third native device reported three successful cold starts, completed all
twenty frozen temperature tasks, and retired the application. Independent
replay confirmed 20/20 task outcomes; the suite took 1,107.357577 seconds.
The host's final collector then rejected the six application/checker unit
files because its inventory still allowed only one start's two files. This
exception skipped the public readiness-journal copy before host teardown.
The full native gate remains failed. Its original result, exact failure and
independent task replay are preserved in
`organism/evidence/first-lineage-native-execution-20261017-03`.

The correction keeps the single-start default and supplies a three-start
allowance only for this frozen cold-start profile. It preserves full owner
identity and unit-file integrity checks and rejects a fourth start or a
second individual. Seventy-six focused checks passed. Before/after sources,
the reproduced original failure and test evidence are in
`organism/evidence/first-lineage-cleanup-preparation-20261017-01`.
The fourth bounded native request must repeat the complete original gate.
Its application, weights, release, cases, resource limits and lease bounds
are unchanged. No lineage allocation is admitted by the partial third-run
success.

### Native admission and first trial: October 17, 2026

The fourth corrected native run passed the complete independent artifact audit:
three cold starts, twenty useful tasks, retirement and host cleanup. External
launch/input/bootstrap checks and complete temporary-resource removal also
passed. Its `external-native-admission.json` is the admission record in
`organism/evidence/first-lineage-native-execution-20261017-04`.

The first fresh three-device trial uses that same release and the common
lineage kit `first-lineage-common-kit-20261017-03`. Its single provisioning
request, private enrollment receipts, fixed lease, spending reservation and
input checks are recorded in `first-lineage-trial-20261017-01`. The initial
lease expires at 2026-10-17 09:25:05 UTC and will not be extended. Closed
additional CPU usage was 3.02978892 instance-hours before reserving twelve
instance-hours for this allocation. No autonomous lineage pass is yet claimed.

### October 17, 05:42 UTC: first lineage startup failure and bounded repair

The first three-host attempt stopped before any receiver was enabled. All three
root-network inspection commands returned SIGSEGV; failure archives and exact
AWS cleanup receipts are retained. All nineteen compute-stack resources, nine
disks, three interfaces and workload/expiry permissions are removed. The attempt
used at most 0.195409 instance-hours (combined closed upper bound 3.225198).
The owner-only inspection now retains and checks bounded raw RTM_GETLINK replies
for the bridge and tunnel. It preserves the original network admission conditions
and passed 47 focused tests. The next attempt has new identities, one bounded
provisioning request, a fresh fixed lease of at most four hours, and the unchanged
release, device package, native gate and task suite. Its twelve-hour reservation
keeps the combined upper bound below thirty hours. Existing temporary inputs are
transferred by exact hash, then the failed trial storage and private keys are removed.
The three successful lineages and two-hour endurance requirement remain pending.

### October 17, 05:59 UTC: forwarding inspection correction

The second fresh lineage attempt reached the complete network readback, then all three hosts rejected the tunnel own local bridge entries as unexpected remote forwarding. No receiver was enabled. Full archives are preserved; all nineteen stack resources, nine volumes and three interfaces are verified removed. Temporary input storage remains only for preservation and transfer to the next fresh trial. The owner-only parser now validates both declared remote peers and both observed local entries, rejecting extra destinations, remote overrides and interface mismatches. All three complete native records reconstruct and 49 focused tests pass. The release, model, device package, birth allowance, lease limits and milestone criteria are unchanged. Closed additional CPU instance-hours are conservatively bounded at 3.512199234444444444444444444.


Update 2026-10-17 06:11 UTC: The third lineage allocation was cancelled before any launch permit or owner enable after source review found that the seed observer omitted the receiver mount join already proved in the single-device native run. No organism or descendant started. Its three machines, nine disks, three interfaces, network, expiry, four roles and three profiles are verified removed. Temporary input storage awaits verified copying and evidence preservation. The narrowly scoped seed observer and independent seed audit now enforce the receiver mount view; 47 focused tests passed, with earlier failed test attempts retained. The trained release and all development checks are unchanged. Two initialization failures and one pre-permit cancellation are preserved; there are still zero completed native lineages. Closed CPU usage is conservatively bounded at 3.627620884444444444444444444 instance-hours. A fresh fourth allocation is being prepared under the existing limits.


2026-10-17 06:29 UTC: Fourth allocation passed stopped-receiver initialization but seed bootstrap failed before any birth with `OSError: [Errno 113] No route to host`. Complete native archive shows identical seed and bootstrap org0 MAC d2:e7:25:40:09:04. No descendant enabled. Full seed failure archive is retained with verified SHA-256, all 19 worker resources and nine volumes are removed. Closed additional CPU hours upper bound is 3.974213563888888888888888888. Explicit distinct addresses for all four lab endpoints and their veth peers, plus paired-port/address admission checks, pass 49 focused tests. Fifth allocation uses common kit 07 and the same model release; still zero passing lineages.

### October 17, 08:00 UTC: asynchronous upload correction

The fifth allocation failed before any owner enable because bridge-port learning
added an unexpected dynamic forwarding entry. Its failure evidence and complete
cleanup are retained. The bridge correction passed 51 focused tests and all
three receivers in the sixth allocation passed native network initialization.

The sixth seed started and passed all twenty frozen tasks. Independent replay
verified the seed's literal local-model establishment choice for the correct
child. Its model upload then failed twice at the asynchronous sender's fixed
120-second outer deadline. That deadline cannot carry the 2.7 GB signed model
under the unchanged bandwidth limit. The synchronous sender already uses a
size-based allowance; the asynchronous path did not use it.

The isolated correction applies that existing allowance to the payload only.
Progress cannot restart the clock; control-message, idle, original birth,
owner-stop and resource limits remain enforced. The original source fails the
new regression case and the candidate passes 88 focused tests. The sixth
allocation remains failed and its evidence must be preserved before cleanup.

Once all its instances are verified terminated, apply the reviewed correction
and rebuild the complete package with the exact same trained weights and pinned
runtimes. A new release requires three cold starts and all twenty native tasks
before new lineage admission. Three fresh passing lineages must use that same
new release, with the final descendant pair completing two hours after seed
disconnection. The standing cost and CPU-hour ceilings, original failure
history, and broader deferred qualification requirements continue to apply.

### October 17, 10:10 UTC: reproduction clock admission

The replacement release passed its complete native gate. Its seventh lineage
seed also passed all twenty tasks and the independent archive audit, but the
child expired the original twenty-minute readiness wait without a birth. The
seed authenticated twenty available-child observations and one successful local
readiness check, then recorded no reproduction intent or model decision.

A real-TLS regression using a retained fractional discovery clock reproduces
that outcome. Subtracting the issue clock from a deadline formed by adding 1200
can produce `1199.9999999999998`; exact comparison rejected a valid proposal.
The correction compares the deadline directly to `issued_clock + 1200`, with
no tolerance, reset or extension. The matching independent transport audit uses
the same original-deadline relation. Sixty-eight focused tests pass, including
fractional-clock transfer reconstruction and rejection of a one-step deadline
increase. The exact rejected native proposal clock was not retained, so this
native causal attribution remains an inference.

All seventh-trial compute resources are removed and both complete host archives
are preserved. Finish owner-evidence, temporary-storage and enrollment closure.
Rebuild the package with unchanged weights and runtimes, and repeat three cold
starts and all twenty native tasks before another lineage. The remaining three
passing lineages and two-hour endurance gate, original limits and retained
failure history are unchanged.

### Owner startup observation correction, October 17 at 11:05 UTC

The seventh lineage is fully closed. Native execution 06 of the corrected
application independently reconstructs two successful cold starts, followed
by a third exit before application, model or checker startup. Its complete
result and owner archives are preserved with verified full checksums and
versions. All compute and temporary-storage resources are removed.

The exact native guard decision was not captured. A real persisted-state
regression nevertheless demonstrates a relevant race: sampling the boot clock
before reading the state transaction can reject a legitimate start when a
concurrent observer advances the saved clock. Sample time after reading that
transaction snapshot. Preserve every owner, boot, token, status, event-chain
and original-deadline check. Record a bounded denial message in the existing
owner log. Separately, report an exhausted process-start allowance immediately
when no process claim remains; preserve pending starts and remaining retries.

The applied corrections pass 112 focused tests. They change only the owner
guard and observation code, plus focused tests. Reuse the exact existing signed
application capsule and rebuild the owner kit. Native execution 07 must pass
three fresh cold starts and all twenty cases before trial 08 may allocate
compute. Trial 08 currently has only public inputs staged; rebind its owner
kit to the new native run before enrollment or launch.

Closed additional CPU use is bounded at 10.430395155 instance-hours. The
30-hour ceiling, conservative $367 project reservation, standing ceiling below
$400, original birth allowance, three passing lineages and two-hour endurance
requirement remain unchanged.


### October 17, 12:00 UTC: native gate admitted; fresh lineage execution

Native execution 07 passed final external admission for release
`00c09aa2aae5f322b2b74358a65153821f99b5c29ef8a99e55170d0e14e33b8d`: three cold starts,
all twenty frozen tasks, resource compliance, retirement and complete temporary
resource cleanup. Both result and owner evidence are preserved with exact
versions and full checksums. The native source records are now frozen.

Trial 08 is rebound to common kit 11 and this admitted release. Prepare fresh
enrollment and verify all twenty-one input objects and actual deployed controls
before publishing permits. Complete seed, child and grandchild task checks,
with seed termination and interface removal before grandchild enable. No
lineage has passed yet; three fresh passes and final two-hour endurance remain.

Closed additional CPU use is bounded at 11.00117957833333333333333333
instance-hours. The first three-device reservation of at most twelve hours
fits the thirty-hour ceiling. Recompute remaining capacity after every closure;
never extend an active lease. All existing limits and deferred Q1-Q10 gates
remain unchanged.

### October 17, 14:14 UTC: first lineage passed; repeat and endurance preparation

Trial 08 passed final external admission at 13:50:24 UTC: all sixty frozen
tasks, both complete peer transfers, seed removal before grandchild eligibility,
and complete temporary-resource cleanup. The result and owner archives are
preserved with exact versions and full checksums. This is the first of the
three required fresh lineages on release `00c09aa2aae5f322b2b74358a65153821f99b5c29ef8a99e55170d0e14e33b8d`.

Trial 09 uses fresh enrollment and the same admitted release. Its three
instances were allocated at 14:00:40 UTC after renewed capacity admission.
Actual deployed controls and input identities passed verification before
permits were published. All receivers initialized stopped. Only the seed is
enabled; its twenty-case observation is in progress. Its original all-device
deadline remains 17:53:35 UTC.

For the final endurance allocation, prepare an earlier fixed seed shutdown
in the original compute request, enforced by both its guest timer and an
additional independent schedule. Preserve both descendant four-hour leases,
the original all-device shutdown, the unchanged application and the original
twenty-minute birth allowance. The shorter seed reservation includes a
600-second shutdown allowance. All actual lifetimes must still be reconciled
at closure. This candidate has passed 31 isolated preparation tests but has
not been applied or admitted for allocation. Apply it only after trial 09
passes external closure and the original owner checks pass regression tests.

The exact release recovery index and its two updated small payloads are
preserved in `first-lineage-release-recovery-20261017-01`. Eight matching
component versions and the fully verified trained-model member of the existing
export archive complete the recovery set. Assemble and verify the complete
capsule before enrollment; these storage locations are not runtime dependencies.
Retained evidence is 49,661,486,035 bytes across 392 versions, below the
50,000,000,000-byte limit. Two lineage passes and final endurance remain.

### Final endurance allocation admitted, October 17, 2026, 15:44 UTC

Trial 09 passed external admission at 15:41:18 UTC. Together, trials 08 and
09 establish two complete passing lineages on release
`00c09aa2aae5f322b2b74358a65153821f99b5c29ef8a99e55170d0e14e33b8d`.
All sixty tasks and both complete peer transfers passed in each trial, with
seed disconnection before grandchild eligibility and all temporary resources
removed. Closed additional CPU use is bounded at
19.29481898166666666666666666 instance-hours.

Trial 10 uses fresh identities and the same carried release. Thirty-one
candidate checks and twelve applied owner regression tests verify its earlier
fixed seed expiry. Its original request contains both the seed guest timer
and independent seed schedule for 17:13:04 UTC, while both descendant guest
timers and the all-device schedule retain 19:43:04 UTC. The deadline was fixed
at 15:43:04.033889 UTC, after all twenty workload inputs were verified; the
owner template was then uploaded and all twenty-one objects reverified before
the sole compute request at 15:44:20 UTC. With the additional 600-second seed
shutdown allowance, its reservation is 9.666638425833333333333333333
instance-hours and the combined bound is 28.96145740749999999999999999,
below thirty. No active lease can be extended. The third lineage and two-hour
endurance remain unpassed until their actual evidence and cleanup are audited.
