# Read-only embodied tools

[Migration and validation report, 2026-10-14 (Chinese)](MIGRATION_20261014.md)

[简体中文](README.zh-CN.md)

This package adds eight opt-in VLM perception tools to the native RoboTwin
toolkit. It requires no adapter checkout. Without `--tool-plugin`, neither the
tools nor their planner guidance are loaded; LingBot, Joy-Brain, atomic actions,
native success checks, memory settings and evaluation budgets retain their
existing behavior.

## Package layout

| File | Responsibility |
|---|---|
| `definitions/*.py` | Each tool's `PROMPT`, `SPEC`, parser and `TASK` registration |
| `tasks.py` | Shared input validation, parser dispatch and prompt fingerprints |
| `server.py` | One Transformers/PyTorch Qwen3.5 backend behind a WebSocket service |
| `client.py` | Checkpoint/protocol/prompt handshake and inference requests |
| `tools.py` | Saved observations, coordinate conversion results and artifact ownership |
| `plugin.py` | Native toolkit registration, planner guidance and source fingerprints |
| `probe.py` | Recorded-image inference or exact external-planner receipt check |
| `suite.py` | Reference-data smoke checks for all tools and the three depth modes |

`../tool_plugin.py` owns the optional plugin lifecycle, CLI arguments and invocation
accounting. Native `robots.robotwin.evaluate` forwards the same options to workers,
freezes source identities and adds per-tool counts to its existing task reports.
VLA and atomic actions are not replaced by the plugin. The NumPy wire codec is
reused from `rpent.utils.msgpack_numpy`.

## Tools and result contracts

| Tool | Result collection | Interpretation |
|---|---|---|
| `grounding_2d` | `detections` | Object boxes in normalized 0..1000 and source-image pixels |
| `affordance_2d` | `regions` | Interaction-region boxes, not guaranteed grasp poses |
| `pointing_2d` | `points` | Normalized/pixel points and integer `[row,column]` addresses |
| `trace_2d` | `trajectories` | Ordered image-plane end-effector keypoints |
| `trace_3d` | `trajectories` | Manipulated-object positions in camera coordinates, meters |
| `grounding_3d` | `detections` | Camera-frame object boxes, dimensions and rotation encoding |
| `video_grounding_3d` | `detections` | Object boxes relative to the last selected frame's camera |
| `depth_estimation` | `depths` | One camera-to-object/marker/point metric distance, not a depth map |

All tools read saved observations and do not advance state. Use exactly the
returned `step_idx` and `view` when passing a 2D pixel address to
`sample_world_xyz`. Box centers are deterministic derived coordinates, not a
second model prediction or a proven contact point. Three-dimensional predictions
are not automatically transformed into the robot's world frame.

`success=true` means inference and output parsing completed; `found` records
whether the parsed prediction collection is nonempty. Neither flag is official
task success. The planner receives parsed data plus provenance, not `raw_text`.
`<episode>/embodied_tools/<request_id>.json` also preserves the model text; input
and overlay PNGs are adjacent. Plugin attempts, including errors, are recorded
in `<episode>/tool_plugin/events.jsonl`.

## Environments and server

Use the existing RoboTwin agent/environment installation for native CLI workers.
The lightweight `embodied-tools` extra supplies Pillow, WebSockets and MsgPack.
Keep the model backend in a separate CUDA environment: `embodied-vlm` adds
Transformers 5.2+ and PyTorch. Do not mix it into a pinned LingBot environment.
The operator must select a PyTorch/CUDA build supporting the actual GPU; these
extras do not certify an RTX 5090 driver/runtime combination.

Run commands from this repository's root. Set `VLM_PYTHON` to the model
environment's Python and `VLM_CHECKPOINT` to a Qwen3.5 multimodal checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLM_PYTHON" -m robots.robotwin.embodied.server \
  --checkpoint "$VLM_CHECKPOINT" --port 19194
```

This is not vLLM or an OpenAI-compatible HTTP endpoint. It uses bf16, SDPA,
greedy decoding, disabled thinking and up to 1024 generated tokens. One shared
model serves all eight prompts; inference is serialized with a lock. The
unauthenticated service binds only to loopback. Start it separately and keep it
running while workers borrow its endpoint; worker cleanup closes connections,
not the shared server. Optional imports keep Torch out of the agent process.

For compatibility, omitting the checkpoint retains the migrated implementation's
local default. Always pass an explicit checkpoint on a different machine; a
matching path, protocol and prompt fingerprint are checked by each client.

## Native evaluation

The existing native Joy-Brain backend remains selected by `--vla-backend
joybrain`. Configure the Joy-Brain source/checkpoint, RoboTwin assets, policy
Python and provider credentials as documented in the native Joy-Brain guide.
Example with two GPUs and one worker per GPU:

```bash
"$RPENT_PYTHON" -m robots.robotwin.evaluate \
  --tasks all --episodes-per-task 1 --gpu-ids 0,1 --workers-per-gpu 1 \
  --vla-backend joybrain --vla-python "$VLA_PYTHON" \
  --vla-model-path "$VLA_CHECKPOINT" --joybrain-repo "$JOYBRAIN_REPO" \
  --robotwin-assets-path "$ROBOTWIN_ASSETS" \
  --model "$RPENT_MODEL" --base-url "$PLANNER_BASE_URL" \
  --tool-plugin robots.robotwin.embodied.plugin \
  --tool-plugin-endpoint ws://127.0.0.1:19194 \
  --tool-plugin-checkpoint "$VLM_CHECKPOINT" \
  --output-dir "$NEW_OUTPUT_DIR"
```

The in-tree plugin needs no `--tool-plugin-root`. For the unchanged no-perception
baseline, omit the three `--tool-plugin*` options and do not start a VLM server.
Do not change task/seed, strategy, memory or budget settings between comparison
runs. The native evaluator's memory defaults are not the same as historical
`--no-memory --no-reference` runs; set evaluation mode deliberately. A plain HTTP
planner URL also requires the native explicit `--allow-insecure-http` consent.

The native scheduler accepts physical GPU identifiers, so eight available GPUs
can be selected with `--gpu-ids 0,1,2,3,4,5,6,7`. This is a configuration example,
not evidence of an eight-GPU hardware test. Ensure the additional shared VLM
fits alongside any VLA/environment on its chosen GPU. Enabling the plugin does
not alter native action-step accounting or the official success predicate.

The standard evaluator defaults to `30000 + 8000 * workers_per_gpu` MiB free
per selected GPU in `--gpu-layout colocated`, or
`max(30000, 8000 * workers_per_gpu)` in `--gpu-layout split`. Split layouts use
model/environment GPU pairs. Preserve a measured safety margin when setting
`--gpu-start-min-free-mib` and `--gpu-runtime-min-free-mib`; both guards remain
active. These inherited controls are unchanged by the plugin, and accepting
eight GPU IDs alone does not establish compatibility with a particular 5090
configuration or its available memory.

The same three plugin options also work with `robots.robotwin.resident_evaluate`.
This experimental runner supports one task at a time, one resident worker per
GPU and at least two episodes per GPU; it is not a multi-task scene-switching
runner. Each episode gets a fresh perception client and observation state,
while the separately launched VLM stays shared. Its source manifest includes
the plugin's prompts/parsers, and its summary reports `perception_tool_calls`
alongside each episode's `perception_tool_usage`. Native process reuse, scene
reset, memory isolation and resource guards are preserved.

## Isolated checks

```bash
"$RPENT_PYTHON" -m robots.robotwin.embodied.probe detect \
  --image "$OBSERVATION_PNG" --tool grounding_2d --target 'the red cup' \
  --checkpoint "$VLM_CHECKPOINT" --output "$NEW_PROBE_DIR"

"$RPENT_PYTHON" -m robots.robotwin.embodied.suite \
  --checkpoint "$VLM_CHECKPOINT" --data-root "$GROUNDING_DATA" \
  --spatial-data-root "$SPATIAL_DATA" --output "$NEW_SUITE_DIR"
```

The suite covers eight tools, with object/marker/point cases for depth. `--audit-only`
checks reference templates without calling a model. Three-dimensional probes
require the saved camera metadata; video probes require ordered extra images.

Replace `detect` with `planner` and pass `--model`/`--base-url` to check a real
planner's exact receipt. Credentials come from `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`, or the environment variable selected by `--api-key-env`.
Optional `--planner-config` accepts a JSON `planner` section with model/base URL
or corresponding environment names, but does not read credentials from another
application's private files. The suite accepts the same flags with `--planner`.
This sends the probe task and tool output to the selected external provider.
Recorded-image probes are not live robot evaluations or success benchmarks.

Offline tests:

```bash
python -m pytest tests/unit_tests/robots/robotwin -q
```

## Preserved limitations

The migration preserves the existing prompts and prediction contracts. In
particular, empty affordance labels still fail strict parsing; the migrated
client also retains its close-on-inference-error behavior. These known issues
are not silently reclassified as successful predictions. They require a
separately validated behavior change. Model output validity never guarantees
geometric accuracy, collision safety or task success.
