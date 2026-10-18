# Build plan: a self-contained, self-replicating Strands organism

October 13, 2026. Status: revision 2; active Codex goal created at the user's request.

**User amendment, October 14, 2026:** human review is not required for the
training corpus. Use recorded automated replay, target/token reconstruction
and provenance checks for data admission. Preserve experiment freeze,
qualification sealing, spending authorization and all Q1–Q10 criteria.
This replaces the earlier human-review prerequisite without claiming that a
human reviewed or approved any example.

**Model direction: fine-tune a specialist around 3B parameters.** Fine-tuning is a core deliverable, with training data work beginning in Phase 1 and the first supervised fine-tuning run in Phase 2. Existing pretrained models provide the starting weights and comparison baselines. Target approximately 3B total stored parameters, with a narrow 2–4B comparison range; a larger deployment model would require a separate scope decision. The initial deployment target is an 8 GiB device, with 4 GiB investigated as a stretch target. Memory and latency remain measured acceptance conditions.

**Build toward an independently useful population of software organisms.** Each individual carries its model, local inference engine, agent runtime, operating knowledge, and complete reproduction package. Established individuals discover newly available eligible devices, establish descendants there, adapt to supported differences, and cooperate to maintain useful services. Every descendant retains the ability to repeat the process after its parent disappears.

The first deliverable is an isolated research prototype with measurable results. On October 13, the user authorized pursuing this plan through a Codex goal. Begin with the Phase 0 inventory and experiment contract, then continue through implementation and qualification. Hardware purchases and cloud spending require their separately budgeted authorization; unrelated paused projects remain preserved.

**The decisive first milestone is an offline seed → child → grandchild demonstration.** The child must supply the software, make the relevant decisions, and establish the grandchild using its own local inference. A test controller that launches descendants, a hosted reasoning model, or a package repository would invalidate that independence claim.

**What will count as the finished prototype.** A population powered by a qualified, fine-tuned approximately 3B specialist expands onto newly introduced compatible devices while the existing population is healthy; operates on native x86-64 and ARM64 Linux hardware within the declared 8 GiB envelope; performs a useful equipment-diagnostics task; recovers from declared software and peer failures; transfers tested experience; and passes the frozen qualification gates below. This establishes a bounded prototype, not general compatibility or production factory readiness.

**Starting context.** The current workspace README marks previous model projects paused and identifies workbench-assistant research as the active direction. The earlier Strands Harness demonstrates patterns for immutable acceptance contracts, artifact hashes, and separate live-child verification. Its documented parent and live-child path depends on Bedrock, and its existing model qualification does not establish suitability for this project. The AgentCore/SageMaker adapter report likewise distinguishes transport tests from live application qualification. Reuse these engineering patterns selectively; build the new organism in an isolated project area. [L1–L4]

**Evidence informing the approach.** Anthropic's article supports investigating local inference setup and inherited operational notes. Its strongest replication experiment used a separate reasoning server and a harness that launched descendants, leaving our independence requirement unproved. Strands supports local inference and tool calling through Ollama. Multi-platform packaging requires compatible runtime variants. AgentCore supports observing agents hosted outside its Runtime. These are ingredients, not evidence that the proposed organism already works. [W1–W4]

**Initial scope and assumptions.** These are conservative planning choices to freeze in Phase 0 after inventory.

| Item | Proposed initial choice |
|---|---|
| Habitat | An isolated, owner-controlled lab network containing at most six authorized device identities. |
| New-device discovery | Devices advertise an authenticated willingness to accept an agent. Peers discover arrivals without a deployment server or a per-birth operator instruction. |
| Execution substrate | A supported Linux OS, supported CPU, sufficient resources, and a small owner-installed admission/loader component. |
| Complete individual | Full local model inference plus all software and supported-platform artifacts needed to establish another complete individual. |
| First platform | One Linux architecture for the first lineage; native x86-64 and ARM64 for prototype qualification. |
| Model search | Target approximately 3B total stored parameters. Compare at most three pretrained candidates in the 2–4B range as starting points for fine-tuning. Count all required weights, rather than only active parameters in a mixture-of-experts model. |
| First candidate | IBM Granite 4.2-3B, subject to pinned-artifact, native-runtime, memory, and task checks. Its current model card identifies a 3B model with tool calling and an Apache 2.0 license. Prior project checkpoints are not automatically selected. [W6] |
| Training | Required supervised fine-tuning on verified operational trajectories, followed by targeted refinement and quantized deployment evaluation. |
| Device footprint | An 8 GiB device is the initial qualification target; 4 GiB is a separately reported stretch profile. Begin with a bounded 4K-token active context and one concurrent generation per organism. |
| First useful work | Discover an equipment API, configure a read-only diagnostic connection, validate data interpretation, and report evidenced faults. |
| Equipment variation | Simulated temperature/vibration gateways, inspection-system status APIs, and machine-log interfaces, followed by a read-only hardware connection. |
| Adaptation | Configuration and tested procedures using declared tools. Model weights and the reproduction core remain versioned releases. |
| Physical control | Existing equipment controllers retain responsibility for actuation and emergency stops. |
| AgentCore | Optional observation and enterprise integration after offline independence passes. |

The receiving loader supplies a narrow execution entry point: authenticate admission, verify a package, provide a device-local installation slot, and start or stop the accepted package. It contains no model, task knowledge, platform-selection policy, or reproduction planner. This is the declared environmental prerequisite, analogous to requiring a compatible OS. It is not a central host sending agents to devices.

The prototype's autonomy begins after initial enrollment and seed installation. Runtime access to an enrollment authority, package registry, hosted model, external DNS, or online documentation must not be required for an ordinary birth. Research-time training infrastructure is separate from the organism: descendants carry the resulting trained weights and reproduce without contacting a trainer or teacher.

**Architecture.** Keep the model's decisions distinct from mechanisms that must behave predictably.

| Component | Responsibility | Location |
|---|---|---|
| Local model and Strands loop | Interpret device observations, select supported procedures, diagnose failures, and decide when evidence is sufficient. | Inside every organism. |
| Reproduction controller | Execute bounded, recorded transitions from discovered device to independently verified peer; handle interruption and retries. | Inside every organism. |
| Complete release package | Model/tokenizer, runtime variants, dependencies, schemas, device procedures, verification assets, release manifest, signatures, and license records. | Retained by every organism. |
| Local state | Identity reference, task state, device observations, peer knowledge, experience records, and operation journal. | Inside each organism; selected records replicated. |
| Device adapters | Obtain equipment observations and apply supported diagnostic configurations through typed interfaces. | Inside every organism; selected by local role. |
| Membership and experience exchange | Share peer observations and versioned evidence with reachable peers. | Peer-to-peer, without a permanent leader. |
| Local enforcement and supervision | Enforce admission, resource limits, single ownership, process lifecycle, and operator retirement. | Small conventional components on each device. |
| Independent evaluator | Observe results, introduce devices/faults, and reconstruct evidence; never perform an organism's deployment actions. | Test infrastructure, outside the organism. |
| Optional AgentCore exporter | Export approved telemetry when connected; remain bounded and nonblocking when disconnected. | Optional organism component. |

Use the workspace's Python/Strands conventions where they fit, with an isolated dependency lock for this project. Start with a packaged local inference provider supported by Strands; Ollama is the first integration candidate. Freeze its model template, tool protocol, binaries, and dependencies after a real offline check. The current workspace's macOS Harness lock is evidence of an earlier environment, not a Linux portability guarantee. [L3, W2]

Use a portable, signed release bundle that can carry both platform variants and all required model assets. A normal platform-specific image pull may retrieve only the variant for the current machine; explicitly verify that each descendant also retains the other supported variant. Packaging details must satisfy this requirement regardless of container format. [W3]

Use embedded local storage and versioned records before adding a distributed database. Established systems demonstrate gossip-based membership, but this plan does not assume gossip establishes exclusive ownership or strong consistency. [W5]

**What the approximately 3B model will learn.** Train the decision-making behavior needed by the organism: interpreting device capabilities, selecting resource-compatible supported procedures, using exact tool arguments, updating a diagnosis after an error, resuming partial work, verifying that a descendant is useful, and writing concise evidence-backed handoff records. Include the correct behavior for unsupported devices, exhausted budgets, and operator retirement. The runtime continues to enforce identity, permissions, and resource limits independently of model predictions.

The training examples must contain the available tool schemas, observations, actual tool results, and correct next actions through completion. Manuals can provide reference material, but a corpus of explanations about deployment would not demonstrate operational competence. TRL supports supervised fine-tuning with tool-call and tool-response messages; PEFT supports adapter-based training, including QLoRA when GPU memory warrants it. [W7–W8]

Begin with supervised fine-tuning through LoRA; choose QLoRA if the measured training footprint requires it. Plan on a compatible, separately budgeted training GPU, not on training aboard the target devices. Start with 200–500 verified trajectories to validate the protocol and training pipeline, then expand coverage from measured failures. This count is a pilot allocation, not a claim that it will be sufficient.

Use automatically verified examples and, where approved, stronger research-time teacher models to propose trajectories. Human review is not an admission prerequisite under the October 14 user amendment. Admit positive examples only after their actions and completion claims pass the independent lab checks. Keep provenance, usage terms, and actual outcomes with each example; a teacher's confidence is not an outcome label. Preserve any applicable workspace data-provenance requirements.

**Train, export, and measure the actual small-device release.** Compare the selected stock model and the fine-tuned model under matched conditions, then compare the trained reference weights with their intended approximately 4-bit deployment export. Record any adapter merge, quantization method, tokenizer/template change, and loaded artifact identity. Training-time low-bit formats and a deployment runtime's low-bit format are separate artifacts; validate the conversion. A quantized release must still pass the full lineage, task, and control gates. [W8]

At an idealized four bits per parameter, 3 billion parameters represent approximately 1.5 GB of raw weight values (about 1.40 GiB). This arithmetic excludes scales, metadata, tensors stored at higher precision, inference state, working buffers, the agent runtime, and the OS. Active context also consumes memory; the initial 4K context is a product constraint to test, not permission to truncate an important tool result or training target. Use explicit persistent state and relevant local references to keep requests bounded. [W9]

**Independence and containment are part of the design.**

- One active organism is admitted per authorized device slot. A finite set of device-bound admission identities enforces the six-device lab limit even during a network partition. A generation counter alone is insufficient to bound a branching population.
- Every instance has its own identity. Children inherit the release and suitable knowledge, not parent credentials, signing authority, or ownership of an existing task.
- The release authority is offline during ordinary reproduction. Existing release signatures are verified locally; an agent cannot expand its admission rights or sign a modified core.
- Each peer may have one child establishment in progress. Device-local ownership prevents two parents from activating competing instances on the same target.
- Operator retirement is persisted locally and takes precedence over automatic recovery. Peers must not recreate an intentionally retired instance.
- During partitions, local diagnostic work may continue. Exclusive changes require the target's ownership check; a remote suspicion of failure cannot authorize conflicting ownership.
- A fleet-wide retirement message cannot be assumed to reach disconnected devices immediately. Qualification must distinguish local stop, eventual propagation after reconnection, and expiry of a predeclared mission authorization.
- Model outputs and retrieved device descriptions are data presented to fixed tools. They cannot modify the evaluator, grant new permissions, change the population cap, or introduce an undeclared installation route.

These controls allow automatic reproduction inside the declared habitat. They do not depend on a human approving each eligible arrival.

**Phase 0 — freeze the experiment and inventory. Estimated effort: 2–3 engineering days.**

Produce the supported-device matrix, baseline task suite, resource contract, dependency/license inventory, and evaluation plan.

1. Inventory available hardware and environments read-only. Identify what can be reused and what must be borrowed or obtained.
2. Define the exact loader/OS prerequisites and separate them from the software each organism must carry.
3. Select up to three current approximately 3B pretrained candidates with compatible fine-tuning, local inference, and redistribution terms. Check Granite 4.2-3B first; pin revisions and compare the same quantization/context profile. Measure actual peak memory, tool correctness, startup, and bounded diagnostic latency.
4. Create 12 environment profiles. Reserve eight for development and four for blind qualification, with at least one entire firmware/API family withheld. Both CPU architectures are development targets; novelty concerns supported device variations, not execution of unknown instruction sets.
5. Freeze success checks, operator-control cases, test budgets, retry rules, evidence schemas, training-data splits, and the stock-versus-trained comparison before optimization.

Exit: a small machine-readable experiment contract and a human-readable decision record, with every cost-bearing action identified. A missing device or model incompatibility changes the proposed setup before trials, not the definition of a passing trial afterward.

**Phase 1 — build the baseline individual and training environment. Estimated effort: 4–6 engineering days.**

Package the selected approximately 3B stock model, local inference, Strands loop, fixed tools, state storage, supervisor integration, and release evidence. Give it one useful diagnostic task. This stock version supplies the integration baseline and data-collection environment; it is not the planned final specialist.

The model reads real tool results and produces structured actions. It should select and verify a supported configuration rather than generate unrestricted installation commands. The supervisor manages process failures and resource ceilings; ordinary discovery events should not require continuous model inference.

Build a complete manifest with hashes and bytes for model, tokenizer, runtime variants, dependencies, procedures, tests, and licenses. Separate the immutable release from instance state. Cold-start with empty caches and external connectivity unavailable.

Start collecting verified successful and failed trajectories from the development fixtures. Verify the native tool-call representation, assistant training masks, complete action targets, context bounds, and family-based split before the first optimizer run. Keep cloud teachers out of the organism itself.

Exit: six successful cold starts, correct local tool use, the complete package present, no required remote service or hidden download, and an audited initial training corpus. Scripted model responses may test plumbing, but do not pass the live-model gate.

**Phase 2 — fine-tune the first 3B specialist and demonstrate reproduction. Estimated effort: 8–12 engineering days.**

Run the first supervised fine-tuning experiment on the audited development-only corpus. Compare the checkpoint against its exact stock parent, export a supported low-memory candidate, and verify that its actual tool behavior survives export. Preserve optimizer evidence, data lineage, full checkpoint identity, and the stock comparison. A failed training candidate remains a failure; do not silently replace it with a larger model or count a stock-only demonstration as trained-model success.

Introduce an eligible device only after the seed is healthy. The seed must notice it through the declared discovery interface, inspect its capabilities, select the supported setup, supply the full package, and establish the child. The child must pass its own task checks.

Disconnect the seed and introduce another eligible device. The child must establish the grandchild using its own inference and local package. No parent proxy, grader-triggered activation, shared host model server, or registry is allowed.

Exercise duplicate arrivals, interrupted package delivery, and two parents attempting the same target. Record every attempt, including failed partial births; retries remain inside the original operation budget.

Exit: three complete development lineages using the trained candidate, with no operator actions after introducing devices, plus passing interruption/ownership checks and the matched stock comparison. This is the first end-to-end milestone and the point to decide whether the concept warrants broader work. Earlier stock-model lineages remain useful integration evidence and are labeled separately.

**Phase 3 — add heterogeneity and useful specialization. Estimated effort: 5–8 engineering days.**

Add the second native CPU architecture and the development equipment profiles. Each individual retains both supported runtime variants while running its local one. Extend the diagnostic task to capability discovery, API/firmware interpretation, unit checks, data freshness, and evidence-backed fault reporting.

Run native cross-platform reproduction in both directions. Emulated architecture checks can detect packaging errors, but do not qualify native memory, latency, or fault independence.

Compare three systems with the same non-model software, tools, admission rules, budgets, and equipment information:

- A conventional baseline selecting fixed recipes.
- The stock approximately 3B model running the complete agent.
- The fine-tuned approximately 3B model running the same agent and intended low-memory export.

Freeze challenge cases before comparing them. Report both the contribution of model reasoning over fixed recipes and the contribution of fine-tuning over the stock model. Successful replication alone does not establish either improvement.

Exit: native cross-platform lineages, complete retained packages, and measurable task performance on development variations. Proceed to the blind gate only after the release is frozen.

**Phase 4 — add cooperative recovery and inherited experience. Estimated effort: 4–6 engineering days.**

Introduce versioned peer observations, durable task records, and bounded exchange of procedural evidence. Preserve uncertainty: a peer being unreachable is an observation, not proof that its equipment failed.

An experience record contains the relevant device/software conditions, observations, attempted actions, result, independent checks, and known limitations. Recipients treat it as a candidate procedure and validate applicability. Device secrets and unnecessary raw equipment data remain local.

An event is acknowledged as replicated only after the required independent copies exist. During isolation, an individual can retain local events without claiming network durability. Retirements and ownership changes use explicit records and device-local enforcement.

Test local process recovery, interrupted births, loss of the original seed, failure of a later peer, network partitions/rejoins, stale observations, and invalid inherited procedures. Software recovery must restore the measured diagnostic service; a log saying “recovered” is insufficient.

Exit: all declared recovery/control cases pass; experience records survive parent loss; no duplicate active owner or revival of a retired instance occurs.

**Phase 5 — refine the trained 3B specialist and qualify its footprint. Estimated effort: 5–10 engineering days.**

Use the stock-versus-trained results and failures from Phases 3–4 to separate reasoning, tool protocol, context management, package compatibility, and environment defects. Repair conventional integration defects before adding examples intended to change model behavior.

Expand the Phase 2 corpus with verified diagnosis, recovery, resource-aware decisions, and handoff examples covering the new development environments. A trajectory includes observations, structured actions, actual results, wrong hypotheses, corrections, and independent completion evidence. Set corpus and compute ceilings before this refinement iteration; sample count alone is not a readiness metric.

Group data by device/API/firmware family. Keep blind qualification families, evaluation outputs, and descendants' qualification experiences out of training. Include correct decisions to wait, decline an unsupported configuration, stop after an exhausted budget, and report incomplete evidence.

Run the targeted supervised refinement and compare it with both the earlier trained checkpoint and the exact stock version through matched providers and templates. Track weights, quantization, tokenizer, context, prompts, and tools so training effects are not confounded with serving changes. Preference training is a later method decision only if the observed errors and verified preference pairs justify it; it is not needed to call the planned supervised work fine-tuning.

Select among trained checkpoints using development gates, then verify the chosen approximately 4-bit export on both native architectures. Measure the whole organism on the 8 GiB target, including cold start, sustained use, staging a descendant, and peak context. Characterize a 4 GiB profile separately; require clean refusal if a device cannot meet its declared working-memory allowance.

Research-time training may use separately budgeted compute. Production individuals must still infer and reproduce locally. Descendants do not autonomously rewrite their weights or the reproductive core in this prototype.

Exit: a trained checkpoint with demonstrated benefit over stock, preserved protected behaviors, and a measured deployment footprint, or an explicit finding that the small-model training goal remains unmet. Retain an earlier trained checkpoint if the refinement regresses. Training completion alone cannot pass the gate, and a stock-only fallback does not satisfy this revised project objective.

**Phase 6 — qualify the complete population and add optional visibility. Estimated effort: 3–5 engineering days, including a 72-hour observation window.**

Freeze the full release containing the selected trained approximately 3B checkpoint, run the blind qualification suite, and reconstruct its results from independent records. After the offline run passes, optionally connect an AgentCore exporter and repeat connection-loss cases. AgentCore observation of externally hosted agents is documented; it does not make AgentCore a required runtime dependency. [W4]

Produce a qualification report, release manifest, software/license inventory, install/uninstall and retirement instructions, and a demonstration recording that includes a new-device arrival, a descendant birth after parent loss, and service recovery.

Exit: all required gates pass on the same final release. Any failed gate remains visible. A corrected release gets a new identity and the relevant full rerun; passing cases are not cherry-picked across releases.

**Qualification gates.** These are proposed prototype thresholds. Phase 0 may revise them transparently for the measured lab envelope before optimization; afterward they remain fixed for that experiment version.

| Gate | Passing requirement |
|---|---|
| Q1 — complete local individual | Six of six cold starts from the declared substrate, with empty caches, full package verification, real local inference, and no undeclared external request. |
| Q2 — autonomous growth | Ten of ten seed → child → grandchild trials pass. New devices arrive while existing peers are healthy; the seed is disconnected before the grandchild's opportunity appears. |
| Q3 — generations and hardware | Three of three native four-generation chains pass, spanning x86-64 and ARM64. At least one chain starts on each architecture. Every generation retains the complete two-platform reproductive package. |
| Q4 — useful adaptation | At least 22 of 24 blind diagnostic/commissioning trials pass across four withheld environment profiles; all 12 protected known-task regressions pass. A complete birth includes useful-task success. |
| Q5 — model contribution | On the same frozen 24 challenge trials, the adaptive agent achieves at least three more successes than the fixed-recipe baseline. Report all outcomes and timeouts. If this fails, describe the result as autonomous provisioning without established model benefit. |
| Q6 — recovery and control | All 12 declared fault/control scenarios below pass, with zero duplicate active owners, boundary violations, revived retirements, or lost events previously acknowledged as replicated under the tested fault assumption. |
| Q7 — useful inheritance | On 20 paired held-out experience-transfer trials, the inherited-procedure arm has equal or higher success and at least 20% lower median bounded effort per scheduled case. Charge timeouts the full budget; report unsuccessful cases. No new protected-task regression. |
| Q8 — sustained small-device operation | A 72-hour run of the trained deployment export on the declared 8 GiB device profiles stays within frozen CPU, RAM, storage, and network limits, maintains useful work, and admits a new eligible device after at least 48 healthy hours. No external service is required. Report the 4 GiB stretch profile separately. |
| Q9 — reconstructable evidence | An evaluator reconstructs every birth, model call, package identity, ownership change, task result, retry, and fault outcome from immutable inputs and recorded evidence. No missing or unexplained activation. |
| Q10 — fine-tuning contribution | On an additional frozen 64-case comparison, the trained deployment export achieves at least four more successes than the matched stock export, or equal/higher success with at least 20% lower median bounded effort per scheduled case. Both use the same tools, prompts, quantization family, context, budgets, and hardware. All protected regressions and Q1–Q9 must still pass. |

For Q4/Q5, repeat each withheld profile six times with predetermined seeds/conditions. The model-benefit comparison is intentionally separate from the autonomy gates: a useful distributed deployment system and a demonstrated need for model reasoning are different results.

Reset local experience and disable cross-trial knowledge sharing for Q4/Q5, so later repetitions cannot inherit earlier qualification answers. Use identical resets and inputs for both comparison arms. These small trial counts are engineering acceptance gates; publish per-profile results and uncertainty rather than claiming broad statistical generalization.

For Q7, freeze an additional 20-case experience-transfer set before the comparison. Match starting states and model/runtime versions; vary only access to prior verified procedural records. These are qualification cases and cannot feed Phase 5 training.

For Q10, create its 64 cases from qualification-only device/API families and seal them in Phase 0. Reset experience between cases and use paired seeds and starting states. The trained-versus-stock comparison runs without inherited qualification notes. Charge failed or timed-out cases their full frozen budget when comparing effort, report per-family success and uncertainty, and preserve all outcomes. These cases also remain outside training.

The 12 Q6 scenario families are: inference-process crash; organism-process crash with supervisor still alive; original-seed loss; later-peer loss with a task in progress; interrupted birth; simultaneous parents targeting one device; partition/rejoin; stale device observation; corrupt package; inherited-note rejection; operator retirement followed by restart; and admission/resource rejection. Test both an inapplicable note and an instruction-bearing note in the tenth family. Test an ineligible identity, exhausted population allowance, and exhausted resource allowance separately in the twelfth. Every subcase must pass; each expected task, ownership, and evidence outcome is fixed in Phase 0.

**Evidence that makes independence reviewable.** Each trial records:

- Experiment/release identity, device identity, parent identity, and lineage generation.
- Actual hardware/OS, loaded model/tokenizer/runtime hashes, context settings, and complete retained package hashes.
- Arrival, discovery, admission, local inference, installation, activation, task-validation, and parent-disconnection timestamps.
- Tool inputs and outputs, budgets, resource measurements, retries, interruptions, and unsuccessful intermediate states.
- Independent network observations that distinguish permitted peer/device traffic from hosted inference, package fetching, and external telemetry.
- A live-child test receipt and evidence that the child, rather than a controller, initiated its descendant's establishment.

The test controller can introduce a device, disconnect a parent, inject a declared fault, and observe results. It cannot issue the reproduction decision, supply a missing artifact, answer a model request, or activate a child. Physical native runs must also remove shared-host inference and storage shortcuts.

**Resource plan.** Final allocations follow Phase 0 inventory; these are sizing assumptions, not claims about available hardware.

| Resource | Planning assumption |
|---|---|
| Development | Existing workstation for packaging, simulation, and evidence analysis. Virtual devices are identified as such in results. |
| Native qualification | Four independently powered devices across two x86-64 and two ARM64 Linux systems for four-generation lineages. Borrow/reuse before procurement. |
| Per device | Target 8 GiB RAM, four CPU cores, 32 GiB free storage, and a supported local runtime for the approximately 3B quantized organism. Qualify actual peak and staging usage. Evaluate 4 GiB as a stretch profile; larger development hosts do not establish small-device performance. |
| Model footprint | Approximately 3B total stored parameters, an approximately 4-bit deployment export, a bounded initial 4K-token active context, and one concurrent generation. Count every mandatory model and runtime in the resource inventory. |
| Training resources | One compatible research-time GPU sized from sequence length, batch/accumulation, and LoRA/QLoRA preflight measurements. Reserve a bounded budget for the first supervised run and one refinement; verify current availability and costs before submission. |
| Network | Isolated reference LAN, nominally 1 Gb/s. No external connectivity required during qualification. |
| Population | At most six distinct authorized device slots, one active organism per slot, and one birth attempt in progress per peer. |
| Initial performance targets | Complete a birth within 20 minutes, recover the declared diagnostic service within two minutes of a confirmed recoverable fault, and keep peak working memory below 75% of assigned RAM. |
| Background work | Healthy idle monitoring should average no more than 5% of one CPU core; transfers and journals have enforced limits frozen from Phase 0 measurements. |
| Storage accounting | Include the running release, the complete reproductive package, temporary delivery/staging space, bounded evidence, and rollback needs. Deduplicated bytes count only where the implementation actually shares them. |
| Cloud | Organism operation remains offline-capable. Fine-tuning requires a planned compute allocation, local or cloud; choose and price it during Phase 0. Optional AgentCore costs are estimated separately. |

The 8 GiB target and 4 GiB stretch target are engineering goals, not claims of measured fit. Measure the complete runtime under the chosen context and workload. If an approximately 3B candidate misses the envelope, investigate runtime overhead, context/state management, quantization, and training efficiency; do not silently enlarge the deployment model or substitute shared inference. Any change to the declared envelope is explicit and versioned before a new qualification experiment.

The 20-minute birth and two-minute recovery targets are engineering targets for the reference lab, not results inferred from the Anthropic article. Keep accuracy and full ancestry evidence even if latency is disappointing.

**Schedule and decision points.** With one engineer working substantially full time, the revised stages total 31–50 engineering days, approximately six to ten focused workweeks, including the first supervised fine-tuning run and one refinement. The first trained lineage should be attempted after Phases 0–2, approximately 14–21 engineering days. Stock integration demonstrations may happen earlier. Hardware access, data-verification effort, difficult portability defects, and additional training iterations can extend elapsed time; these estimates are not production-readiness commitments.

| Decision point | Evidence required | Resulting choice |
|---|---|---|
| After Phase 0 | Approximately 3B candidate measurements, 8 GiB resource contract, training-data plan, and priced compute options. | Freeze the starting model and training method; keep any resource change explicit. |
| After Phase 2 | A real offline grandchild using the first trained candidate, with no parent/controller assistance. | Continue, repair the training/independence defect, or record the unmet milestone. |
| After Phase 3 | Native portability plus fixed-recipe, stock-model, and trained-model results. | Identify the next data and engineering improvements; keep all three comparisons visible. |
| Before Phase 5 | Failure taxonomy, verified new trajectories, and a bounded refinement budget. | Refine the selected approximately 3B specialist without admitting qualification data. |
| After Phase 6 | All gates on one release and independently reconstructed results. | Admit a separately scoped read-only field trial; physical control requires its own qualification. |

**Workspace organization and reuse.** On implementation, create an isolated project area such as `organism/` containing the experiment contract, core runtime, device adapters, package builder, schemas, evaluation fixtures, and documentation. Put generated run evidence under a designated run directory and model artifacts in a declared artifact store; keep them out of source control unless intentionally small fixtures. Use the actual repository/hosting setup discovered at that point, with GitLab-oriented change review.

Reuse the immutable acceptance-contract and manifest-publication patterns in `examples/harness_request_builder.py`, with their relevant tests. Reuse the separation between build success and live-child success in `docs/strands-harness.md`. Adapt evidence reconstruction from prior work only after reviewing its assumptions. Do not carry over a cloud parent, old model selection, implicit credentials, or incompatible platform lock as an invisible dependency. [L2–L4]

**First implementation batch.** When implementation is requested, begin with the following concrete deliverables:

1. A dependency and hardware inventory, approximately 3B base-model selection, 8 GiB target contract, and training-compute estimate.
2. Definitions for device capabilities, admission policy, release manifest, structured model action, action result, and independent validation receipt.
3. Diagnostic fixtures, audited operational trajectories, sealed evaluation splits, and the first supervised fine-tuning recipe.
4. A complete stock-model baseline, followed by the first trained low-memory individual, each tested from an offline cold start.
5. A three-device lab demonstration using the trained candidate for peer-initiated seed → child → grandchild establishment, including the parent's removal.

Make model specialization and device fit part of the core engineering work. Keep AgentCore integration behind the offline-independence gate. The first investment must answer whether a complete individual powered by our trained approximately 3B specialist can establish another that is independently capable of doing the same.

**Source record.** Public references were checked on October 13, 2026. These sources substantiate individual capabilities and constraints; all proposed thresholds and schedule estimates above are our engineering choices.

- **[W1] Anthropic, “AI models may be capable of self-replicating without access to a datacenter,” October 12, 2026.** Relevant to inference setup, the limits of its replication setup, and inherited notes. URL: `https://www.anthropic.com/research/self-replicating-ai`
- **[W2] Strands Agents, “Ollama.”** Local inference, tool calling, and provider configuration. URL: `https://strandsagents.com/docs/user-guide/sdk/model-providers/ollama/`
- **[W3] Docker, “Multi-platform builds.”** Architecture compatibility and per-platform artifacts. The plan does not assume Docker supplies peer admission or reproduction. URL: `https://docs.docker.com/build/building/multi-platform/`
- **[W4] AWS, “Add observability to your Amazon Bedrock AgentCore resources.”** Observability for agents outside AgentCore Runtime; optional cloud integration. Also checked through the AWS Knowledge MCP. URL: `https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html`
- **[W5] Apache Cassandra, “Dynamo.”** Background for distributed membership, failure detection, and consistency distinctions; not a recommendation to embed Cassandra. URL: `https://cassandra.apache.org/doc/latest/cassandra/architecture/dynamo.html`
- **[W6] IBM Granite, “Granite-4.2-3B” model card.** Current 3B starting candidate, tool-calling interface, and published license; this does not qualify our task or local export. URL: `https://huggingface.co/ibm-granite/granite-4.2-3b`
- **[W7] Hugging Face TRL, “SFT Trainer.”** Supervised fine-tuning, adapter integration, and tool-call/response training format. URL: `https://huggingface.co/docs/trl/sft_trainer`
- **[W8] Hugging Face PEFT, “Quantization.”** Adapter training with quantized weights, QLoRA, and format-specific constraints. URL: `https://huggingface.co/docs/peft/developer_guides/quantization`
- **[W9] Hugging Face Transformers, “Cache strategies.”** Context-related inference memory and speed tradeoffs; actual local-runtime measurements remain required. URL: `https://huggingface.co/docs/transformers/en/kv_cache`
- **[L1] Current project direction:** [README](/Users/larstray/Documents/strandsLM/README.md) and [transition record](/Users/larstray/Documents/strandsLM/reports/project-transition-20261010.md). Read as documented state; no cloud-status audit was performed for this planning request.
- **[L2] Existing evidence pattern:** [bound builder](/Users/larstray/Documents/strandsLM/examples/harness_request_builder.py).
- **[L3] Existing integration limits:** [Harness documentation](/Users/larstray/Documents/strandsLM/docs/strands-harness.md) and [isolated dependency lock](/Users/larstray/Documents/strandsLM/requirements-harness.lock).
- **[L4] Existing AgentCore evidence boundary:** [AgentCore/SageMaker integration report](/Users/larstray/Documents/strandsLM/reports/qwen35-9b-agentcore-integration-02.md).
