# RoboTwin parallel evaluation

Run from the RPent checkout after installing its RoboTwin runtime and the
`joybrain` extra. The controller, workers, recorder and dataset converter are
all in this repository. Model implementation is loaded only from the specified
Joy-Brain-vla checkout; weights, simulator assets and credentials remain external.

## Eight colocated workers or four split groups

Provide planner credentials through `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
The model ID must match that provider. No credential helper or private endpoint
is bundled; plaintext provider URLs require explicit `--allow-insecure-http`.

```bash
python -m robots.robotwin.evaluate \
  --joybrain-repo /path/to/Joy-Brain-vla \
  --vla-model-path /path/to/pretrained_model \
  --vla-python /path/to/policy-env/bin/python \
  --robotwin-assets-path /path/to/RoboTwin \
  --model openai:your-model-id \
  --gpu-layout colocated --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 1 --max-active-planners 8 \
  --gpu-start-min-free-mib 30052 --gpu-runtime-min-free-mib 2048 \
  --tasks click_bell --episodes-per-task 1 --max-turns 4 \
  --planner-timeout-s 600 --max-episode-steps 256 \
  --dry-run --output-dir /path/to/new/colocated-plan
```

The example plans eight colocated slots, each with one VLA service and one
environment worker. It plans only one episode; use an explicit task/episode
budget when ready to evaluate more tasks. For four two-card groups and twenty
worker slots, change the layout and concurrency options to:

```bash
--gpu-layout split --gpu-ids 0,1,2,3,4,5,6,7 \
--workers-per-gpu 5 --max-active-planners 20
```

In split mode, the ordered GPU list forms `(model, environment)` pairs:
`0→1`, `2→3`, `4→5`, `6→7`. Only model cards load a VLA; five workers on each
environment card share their group's endpoint. The GPUs must be distinct, the
list must have an even length, and all cards must be available within the
caller's allocation. `--workers-per-gpu` always counts workers per environment
card, not per selected physical card.

The 30052/2048 MiB limits are a calibrated small-card example, not universal
defaults or a guarantee for every checkpoint/task. Re-measure headroom for
the intended assets, task configuration and recording settings before removing
`--dry-run`. Provide valid planner credentials and authorize the image/task
data sent to that provider. A dry run performs no model, simulator or API calls.

Defaults select only `click_bell`, one seed and one worker. The native
RPent prompt is the default; `--strategy atomic-preferred` explicitly opts into
an atomic-transport/placement preference without phase locks or tool quotas.

## Devices and seeds

`--gpu-layout colocated` is the default and keeps one VLA per selected GPU.
For two cards with one worker each, use `--gpu-ids 0,1 --workers-per-gpu 1`. Missing or
duplicate physical cards are rejected; the caller's CUDA visibility is respected.
A borrowed `--vla-endpoint` is supported for one group (one colocated card or
one split pair) and avoids local model loading. The caller must verify the
borrowed endpoint's actual model placement. Inference within a service remains serial; there is no batching
or guaranteed speedup. Provision CPU/RAM as well as VRAM before increasing workers.

GPU admission defaults to `30000 + 8000 * workers_per_gpu` MiB free for a
colocated layout and `max(30000, 8000 * workers_per_gpu)` MiB on each selected
card for a split layout. The runtime guard requires 12000 MiB remaining by
default and checks both model and environment cards. Smaller GPUs require separately
measured model-only, environment-only, and same-card peaks before choosing
`--gpu-start-min-free-mib` and `--gpu-runtime-min-free-mib`. Include a safety
margin in the startup requirement and retain a runtime reserve of at least
1024 MiB; neither protection can be disabled. Both values are recorded in the
plan. Start with one worker, and repeat the measurements when changing the
checkpoint, task configuration, recording settings, or worker count. Specify calibrated limits
again when resuming; they are not inherited as task-semantic settings.

The controller leases all selected model and environment cards, rechecks
headroom after acquiring the leases, and monitors resources while models load
and workers run. Failures stop only owned process groups; they never replace
missing GPUs or kill unrelated jobs.

Tasks are dispatched round-robin by trial. Each task gets distinct seeds: first
its published seeds, then increasing values beyond that list. Added seeds are
not expert-verified and initialization failures are recorded, not replaced.
`--seed N` explicitly selects the consecutive range starting at N instead.
The complete plan is frozen before execution; no automatic action/episode replay.
Scene seeds do not fix the model server's action-sampling RNG; serial and parallel
runs are not guaranteed to produce identical action chunks.

## Startup and worker lifetime

An opt-in, single-task resident-worker experiment is available through
`python -m robots.robotwin.resident_evaluate`; see [RESIDENT_ROLLOUT.md](RESIDENT_ROLLOUT.md).
It retains worker/Env processes while resetting native scenes and isolating
episode clients, conversations and recorders. It does not change the default
evaluator's per-episode process lifetime described here.

Model processes are spawned before the controller waits for readiness, rather
than loading eight models sequentially. Each established connection sends a
lightweight protocol keepalive so an idle model is not discarded while slow
workers initialize. This does not run inference or change the policy RNG.

Every episode still owns a fresh native RPent CLI process and simulator. Worker
slots and model endpoints are reused, **not** the Python worker or scene across
episodes. No persistent-worker/reset optimization is implied by this runner.

Within an environment card, only one worker initializes at a time until its
initial `states.json` observation is published. Initialized workers continue
executing while the next worker starts; independent cards initialize in
parallel. `--worker-startup-timeout-s` defaults to 1800 seconds. Missing readiness
times out that attempt, frees the slot and does not replay the episode. The
overall worker budget also includes the planner timeout and 60 seconds for
finalization.

Private runtime caches default to `<output>/runtime-cache`: each model GPU and
each worker slot has separate Env/VLA Torch, Inductor, Triton, Warp, CUDA and
renderer directories. `--runtime-cache-dir` selects a different root. Slots on
different physical GPUs never share a cache directory. The controller creates
the nested Torch kernel cache directory before startup; no installed package,
system driver or CUDA toolkit is modified. Cache isolation and staggered
initialization reduce startup contention but do not guarantee all native
runtime bugs are fixed.

## Results and traces

`summary.json` and `task_summary.csv` report official successes, timeouts and
initialization errors separately. Every episode has `result.json`, a transcript,
`episode.mp4`, and `trace/`. The trace contains before/after RGB and qpos14 state,
actual executed targets, native counters, tool IDs/arguments, effective VLA
instruction, reward and terminal flags. Unexecuted proposals are separate files.
Atomic tools are recorded at the same native action boundary as VLA execution.

Recording adds rendering and I/O costs. Times are wall times and native decision
indices, not inferred physical control frequency. Failed/incomplete recording
blocks subsequent mutations. Short valid episodes are not mistaken for failed
runs merely because they lack a complete training chunk.

`plan.json` records the layout, physical model/environment pairs, model-server
count and worker capacity. Slot assignments and scheduler results distinguish
`gpu_uuid` (environment) from `model_gpu_uuid`; `ready_ns` records observed worker
readiness when available. Startup timeouts are separate from planner/overall
worker timeouts. Scene success remains the native RoboTwin result, not planner
prose or a successful infrastructure check.

Create `STOP_AFTER_CURRENT` in the output root to stop new dispatches. Interrupts
clean only owned processes. A completed worker does not close its shared VLA
service. Changing source files during a run stops dispatch to prevent mixed-code
results. Never resume an older experiment by silently replacing its code/config.

## Resume with a different worker count

After the parent run drains normally, create a new output directory and use:

```bash
python -m robots.robotwin.evaluate --resume-from /path/to/drained/run \
  --gpu-ids 0 --workers-per-gpu 8 --max-active-planners 8 \
  --output-dir /path/to/new/phase
```

The original task/seed plan, model, checkpoint and budgets are inherited. All
layout choices default to the parent layout; older runs without layout metadata
are treated as colocated. An explicit layout or device/count change is recorded
as mixed concurrency and retains each episode's original assignment. All
finalized attempts, including failures/timeouts, are excluded from new dispatch.
Partial or still-running attempts block resume rather than being replayed.
Worker/model/recorder source hashes must match; a controller-only update may be
recorded separately. Results retain their original directories and phase labels.
`completed_attempts` includes inherited attempts; `phase_completed_attempts`
counts only new work. A restarted VLA has a new RNG stream. Mixed-concurrency
phases are explicitly marked and are not a controlled throughput benchmark.

## Dataset validation

The optional converter uses only the supplied Joy-Brain repository's LeRobot
pipeline. It exports confirmed qpos14 actions, preserves success/failure labels,
and excludes incomplete 16-action windows from sampling without deleting raw tails.

```bash
python -m robots.robotwin.dataset export \
  --trace /path/to/episode/trace --dataset /path/to/new/dataset \
  --checkpoint /path/to/pretrained_model --joybrain-repo /path/to/Joy-Brain-vla
python -m robots.robotwin.dataset validate \
  --dataset /path/to/new/dataset --checkpoint /path/to/pretrained_model \
  --joybrain-repo /path/to/Joy-Brain-vla
```

Nominal dataset fps=1 indexes decisions, not physical Hz. Exported candidates
remain marked for curation; this does not train a model or declare benchmark
improvement. `--allow-validation` is only for engineering smoke-test episodes.

CPU checks: `python -m pytest -q tests/unit_tests/robots/robotwin`.
