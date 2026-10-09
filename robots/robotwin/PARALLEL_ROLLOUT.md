# RoboTwin parallel evaluation

Run from the RPent checkout after installing its RoboTwin runtime and the
`joybrain` extra. The controller, workers, recorder and dataset converter are
all in this repository. Model implementation is loaded only from the specified
Joy-Brain-vla checkout; weights, simulator assets and credentials remain external.

## One GPU, four workers

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
  --gpu-ids 0 --workers-per-gpu 4 --max-active-planners 4 \
  --tasks all --episodes-per-task 10 \
  --max-turns 24 --planner-timeout-s 900 --max-episode-steps 1000 \
  --output-dir /path/to/new/run
```

Use `--dry-run` to validate device selection and save the plan without model or
API calls. Defaults select only `click_bell`, one seed and one worker. The native
RPent prompt is the default; `--strategy atomic-preferred` explicitly opts into
an atomic-transport/placement preference without phase locks or tool quotas.

## Devices and seeds

Each selected GPU owns one VLA service shared by its worker processes. For two
cards with one worker each, use `--gpu-ids 0,1 --workers-per-gpu 1`. Missing or
duplicate physical cards are rejected; the caller's CUDA visibility is respected.
A borrowed `--vla-endpoint` is supported for one selected card and avoids local
model loading. Inference within a service remains serial; there is no batching
or guaranteed speedup. Provision CPU/RAM as well as VRAM before increasing workers.

Tasks are dispatched round-robin by trial. Each task gets distinct seeds: first
its published seeds, then increasing values beyond that list. Added seeds are
not expert-verified and initialization failures are recorded, not replaced.
`--seed N` explicitly selects the consecutive range starting at N instead.
The complete plan is frozen before execution; no automatic action/episode replay.
Scene seeds do not fix the model server's action-sampling RNG; serial and parallel
runs are not guaranteed to produce identical action chunks.

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
