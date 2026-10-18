# First-lineage development operations

Status: the reduced runtime milestone passed on October 17, 2026.
Release `00c09aa2aae5f322b2b74358a65153821f99b5c29ef8a99e55170d0e14e33b8d` passed three native cold starts,
all twenty native diagnostic cases, three fresh seed → child → grandchild
lineages, and two hours of scheduled useful work on both final descendants
after seed removal. All listed temporary cloud resources and local private
enrollment files are removed. No experiment remains running.

The completed runtime report is
`evidence/first-lineage-final-report-20261017-01/README.md`.
Its separate preservation and completion receipts establish final archival
and milestone completion. The conservative closed-lifetime CPU bound is
27.81815310694444444444444444 instance-hours,
below thirty. Project reservations remain $367, below the standing $400
ceiling; these are reservations, not an invoice. A future experiment needs
fresh enrollment and a new bounded allocation. Never extend an active lease.
Broader Q1–Q10 qualification remains deferred.

The complete release recovery index is preserved in
`evidence/first-lineage-release-recovery-20261017-01/preservation.json`.
Recover its exact component versions and verify the complete local capsule
before enrollment. No runtime inference or reproduction step depends on this
recovery storage. Never substitute the older standalone model merely because
its byte length matches; its checksum differs.

This procedure belongs to the reduced
[development plan](../docs/replicate-first-lineage-plan-20261016.md).
The authoritative current gate and evidence pointers are in
[the milestone record](evidence/first-lineage-milestone-20261016-01/status.json).
Do not use this document as a release or qualification certificate.

## Admit one release

1. Require the returned CPU archive to pass the independent host, input,
   numerical-response, and actual-precision audits. Keep the complete archive,
   its checksum, retained S3 version, and reconstructed payloads.
2. Require the trained model to pass every frozen temperature case under the
   same interface and budgets used for the stock and fixed-policy comparisons.
   Keep unsuccessful cases and all original comparisons.
3. Build the signed trained capsule from that admitted model and current
   application. Build the native profile with both `--x86-only` and `--cases`
   pointing at the original frozen twenty-case file. The required profile is
   `organism-native-profile-kit-v3`; a v2 kit lacks this task observer.
   The admitted profile uses `--x86-instance-type c7a.xlarge`.
   Preserve that choice from the signed preparation receipt through the
   launch packet and lineage kit, and verify the actual four-core, one-thread
   CPU topology. The memory limit, task budgets and leases stay unchanged.
4. Complete one actual x86-64 8 GiB trial: three cold starts, all twenty useful
   tasks, measured resource compliance, explicit stop, and persistent retirement.
   Independently audit its archive and join it to the external cloud controls
   and cleanup readbacks before admitting a lineage allocation.
5. Build one common lineage kit from this exact single-device kit. The lineage
   builder rechecks the carried source bytes, export, catalog, and full payload
   inventory. Use that same release for all three fresh trials.

## Before each allocation

Record the exact instance identities, spending reservation, worst-case remaining
CPU instance-hours, retained evidence capacity, and fixed UTC expiries.
If the original request includes an earlier seed shutdown, verify both its
guest timer and independent seed-only schedule, and include the documented
600-second shutdown allowance in the reservation before allocating machines.
Keep at most three active device slots and at most four hours per lease.
Use the standing spending authority and the amended plan's combined limits.
An active lease is never extended.

Use a dedicated temporary bucket and prefix. Verify its owner, encryption,
public-access block, ownership enforcement, and TLS-only policy. Read every
uploaded object's complete SHA-256 and exact size back from S3. Retain the
requests and responses. Check the actual deployed template, bootstrap bytes,
instance and disk inventory, network rules, complete role policies, instance
profiles, and independent expiry before publishing any instance permit.

Prepare fresh enrollment for each trial with `lineage_enrollment.prepare`.
Keep private bundles outside the repository in their restricted temporary
directory. There are three execution identities and a separate owner
seed-transfer identity with no execution slot. Each host receives only its own
bundle; the seed additionally receives the owner's seed-transfer key.
Retain the signed public policy, discovery book, certificates, and identity
receipt. Never copy private keys into the public evidence archive.
Prepare the request with `lineage_launch.launch_packet`. It rechecks the
common kit, verifies the contents of each actual private bundle, and carries
the selected hardware and fixed deadline into the request. This preparation
does not establish the external native admission or authorize an allocation.

The seed's S3 permissions allow the complete package. Each other device receives
only the three carried archives needed for its minimal receiving substrate.
Neither descendant receives its application or model from owner storage.
The instance permits bind all three host identities, each host's three disks,
the common release inputs, its own enrollment bundle, and the original expiry.

## Run each fresh lineage

1. Wait until all three enrolled receivers report their initial stopped state.
   Introduce the seed through its sequence-1 `enable` control. The owner supplies
   the seed package once. Require actual seed readiness and its twenty task
   outcomes.
2. Introduce the child through its sequence-1 `enable` control. The seed must
   discover the available receiver and choose and complete reproduction using
   its own local inference. The observer waits for the child; it has no
   descendant-launch operation.
3. Require the child's full package, distinct identity, original birth
   allowance, own model process, and twenty correct task outcomes. Preserve
   both endpoints' complete transfer and decision records.
4. Give the seed its sequence-2 `finish` control. Preserve its outputs and
   verify its instance is terminated and its exact network interface is absent.
   Record the raw AWS readback and its SHA-256.
5. Only after that disconnection, introduce the grandchild with its sequence-1
   `enable` control, carrying the verified seed-disconnection record. The child
   must establish the grandchild through the same autonomous peer path.
   Require the grandchild's own inference and twenty task outcomes.
6. For trials one and two, finish both descendants through their sequence-2
   controls and preserve their final outputs. Independently reconstruct each
   transfer and the complete trial before counting it as a pass.
7. For the final passing trial, publish sequence-2 `endurance` controls for the
   child and grandchild with the same future start time and the verified seed
   disconnection. The original leases must still contain the two hours and
   cleanup. Run six declared cycles at twenty-minute intervals, each covering
   all five temperature categories. Require correct outcomes, retained model
   evidence, unchanged process identities, and continuous resource compliance.

Controls are exact host-, release-, sequence-, and deadline-bound JSON objects
under `<prefix>/inputs/controls/<slot>/01.json` or `02.json`. Publish a new control with
`IfNoneMatch=*`, verify its checksum, and retain its exact original bytes.
Availability controls only change receiver eligibility. They do not choose
reproduction actions or install descendants.

## Stop and preserve

A normal sequence-2 `finish` ends observation and invokes the owner's existing
retirement cleanup. That cleanup stops observation drivers, persistently retires
the individual, stops its receiver and checker, and verifies the process groups
are empty before collecting bounded public evidence and removing the temporary
runtime mounts and laboratory interfaces. Successful endurance performs this
same cleanup after its two-hour observation; it does not require a third
owner control.

If observation fails, preserve the failure and allow the same cleanup path to
run. If a host is unresponsive or must be stopped immediately, terminate only
the recorded instance through the AWS MCP. Preserve whatever output already
exists; record missing evidence as a failed or incomplete trial. The independent
expiry remains in force until termination and cleanup are verified.

Before removing a temporary bucket, inventory all objects and multipart uploads.
Preserve each required public artifact with verified complete bytes, checksum,
and retained version. Private enrollment bundles are temporary secrets: retain
their public identity receipts, then remove the temporary key copies as part of
the recorded cleanup. Do not archive private keys.

Verify the recorded instances are terminated, all nine trial disks and three
interfaces are absent, and the worker stack, roles, profiles, every expiry schedule, and
network resources are removed. Remove only the dedicated storage stack after
its bucket is safely emptied. Reconcile actual lifetimes and retained versioned
bytes. A local audit alone does not establish external cloud cleanup.

## Completion record

The final result must bind the single-device pass, three fresh lineage passes,
and two-hour endurance pass to one release. Include measured memory, storage,
task latency, transfers, process continuity, failure history, and external
cleanup evidence. Keep the broader Q1–Q10 requirements explicitly deferred.
Until those records exist, the milestone remains incomplete.
