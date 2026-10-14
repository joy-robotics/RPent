# Resident RoboTwin Agent workers / 常驻 worker 试验

`python -m robots.robotwin.resident_evaluate` is an opt-in, single-task experiment.
It keeps one VLA service and one Agent worker per selected GPU. Each worker owns
a persistent EnvServer and runs multiple independent episodes through the native
RPent CLI's API planner, prompts, toolkit, recording and result finalizer.

This is different from `robots.robotwin.evaluate`, which creates a worker for
every episode. The default evaluator is unchanged. Resident mode currently
supports only Joy-Brain, colocated one-worker-per-GPU, one `demo_randomized` task,
and at least two episodes per GPU. Episodes are assigned to fixed round-robin
shards; idle GPUs do not steal another shard's work. There is no automatic worker
replacement or replay after a failure, and interrupted batches are not resumed.

常驻的是 Agent worker / EnvServer 进程及运行时，不是上一条任务的对话、状态或录像。
每个 episode 都重新建立 planner 会话、toolkit、memory 目录、EnvClient、VLA 连接和
记录器。VLA 服务及权重由控制器复用，worker 退出不会提前关闭其他 worker 的服务。

## Reset and ownership

- The resident EnvServer keeps its RLinf environment owner alive. For each new
  seed it refreshes the native task/language prewalk and follows native reset;
  the scene/task is reconstructed, not merely given a new counter.
- Native prewalk metadata must be refreshed: directly calling `reset(seed)` on
  the previous `SubEnv` can reuse its old episode information when generating
  language. The published evaluation language remains authoritative where present.
- Completed scene references and inactive Env CUDA allocations are released
  before constructing the next scene. This does not clear the separate VLA
  process's caches or alter planner/normalization/chunk/success semantics.
- An explicit begin/end transaction rotates the native recorder. Epoch-tagged
  RPC calls reject stale clients, calls before reset, and in-episode resets.
  Recording failure, uncertain execution, or failed reset retires the worker.
- Each reset verifies the requested seed, zero native action count and cleared
  success state. Each episode has separate state files, traces, transcript and
  video. The previous episode is finalized before a new one can begin.

因此，“常驻”不等于零成本 reset，也不代表所有任务可直接切换。当前不会跳过原生
场景初始化、复用错误的语言前置信息、沿用上次成功标记，或把不确定动作自动重放。
GPU 准入、运行期显存余量、进程超时与自有资源清理继续生效。

## Bounded run

First validate consecutive episodes on one allocated, idle GPU. Then use eight
GPUs with the same task/seed and budget as the comparison run. Configure provider
credentials outside scripts; HTTP providers require explicit authorization and
`--allow-insecure-http`. The example only generates a plan.

```bash
python -m robots.robotwin.resident_evaluate \
  --gpu-ids 0,1,2,3,4,5,6,7 --tasks click_bell --episodes-per-task 20 \
  --joybrain-repo "$JOYBRAIN_REPO_ROOT" \
  --vla-model-path "$JOYBRAIN_MODEL_PATH" --vla-python "$JOYBRAIN_PYTHON" \
  --robotwin-assets-path "$ROBOTWIN_ASSETS_PATH" \
  --model "$RPENT_MODEL" --max-turns 12 --max-tokens 2048 \
  --max-episode-steps 256 --planner-timeout-s 600 --joybrain-timeout 600 \
  --reset-timeout-s 600 --worker-startup-timeout-s 1800 \
  --gpu-start-min-free-mib 31000 --gpu-runtime-min-free-mib 2048 \
  --dry-run --output-dir /path/to/new/resident-plan
```

The 31000/2048 MiB values are explicit guard settings for a measured small-card
configuration, not universal guarantees. Keep adequate room for native scene
reconstruction and CUDA workspace peaks. Do not relax the guard to force a run.

Use a fresh output directory without `--dry-run` for the authorized real run.
For a one-card consecutive-episode check, use `--gpu-ids 0 --episodes-per-task 3`.
Do not confuse a successful dry run, interface check, or import with GPU/Agent
acceptance. Keep the initial 12-request budget distinct from unrestricted task
success, including attempts that reach the request limit without `finish`.

## Evidence

The root `plan.json` records the fixed shards, full budgets and source hashes.
`summary.json` aggregates finalized episode results. Each `resident_gpu_<index>`
directory contains a shared `env-server.log`, `worker-result.json`, and separate
episode subdirectories. `resident-runtime.json` and `resident-end.json` record
worker/Env PIDs, epoch, reset status and per-episode counters. `resident.begin`
and `resident.end` are owner lifecycle RPCs, not tools exposed to the planner.

Check identical worker and Env PIDs across consecutive episodes, fresh initial
state/language and counters, non-growing memory, immutable previous traces, real
Agent/VLA/native-action counts, video and complete owned-process cleanup. Report
initial cold startup, subsequent resets, and whole-batch throughput separately.
This experiment does not by itself certify long-run, multi-task stability.
