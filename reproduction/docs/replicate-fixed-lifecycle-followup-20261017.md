# Fixed-policy lifecycle correction

The completed fixed arm of the October 17 cohosted comparison exited before
its first diagnostic. Its adapter reported `inference_ready=false`, correctly
representing the absence of an inference server. The inherited application
loop nevertheless required that flag to be true.

The frozen `control_policy.py`, `control_entry.py`, signed comparison capsules,
and owner archive remain unchanged. The separate
`organism/research/fixed_lifecycle_candidate.py` implements the narrowly scoped
correction: supervise `policy_ready` in both the application and reproduction
loops, preserve failure propagation and graceful shutdown, and continue to
report `inference_ready=false` and zero model calls. The second inference guard
was found during source inspection; the native failure exposed the first.

Four local regression checks pass:

1. The original full application loop reproduces the observed failure.
2. The candidate remains available across a supervisor interval, serves a real
   local diagnostic request, records zero inference calls, and stops cleanly.
3. Loss of policy availability terminates the candidate and closes its resources.
4. The complete policy/discovery/reproduction loop transfers a small fixture
   capsule over real TLS, records zero model calls, and waits for explicit
   fixture readiness before recording completion. The unchanged model auditor
   rejects its no-model trace.

The tests use small, explicitly unused model/runtime fixture files. They do not
demonstrate native capsule installation, a live model, or peer reproduction.
The candidate was not carried by either allocation of the frozen experiment.

For a future comparison, change the fixed application and reproduction imports in
`control_entry.py` to use `FixedLifecycleCandidate` and
`FixedReproductionLifecycleCandidate`, and add that module to both
the carried application and owner inventories in `build_controls.py`. Produce
new signed capsules and a new input freeze. The stock and trained decision
policies, models, prompts, tools, and diagnostic cases need no change.

Rerun all three arms on the same declared hardware and resource limits under
fresh enrollment. Keep the failed experiment and the corrected experiment
separate. Require complete native lifecycle coverage, actual parent-supplied
transfers, useful readiness, and seed absence before grandchild availability;
then reconstruct every outcome. Until those steps pass, this is a tested local
correction candidate, not a validated native deterministic baseline.
