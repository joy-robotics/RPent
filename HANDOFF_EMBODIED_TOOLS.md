# Embodied tools 合并与 5090 接续开发

本文件面向在新节点拉取代码的开发者和 code agent。先读本文，再按链接检查实现；不要把本机 H200 环境路径、组件测试或有限预算验证当作 5090 环境认证或完整 benchmark。

## 1. 代码身份与已完成的迁移

| 项目 | 核验结果（2026-10-14） |
|---|---|
| 远程 | `git@github.com:joy-robotics/RPent.git`，私有仓库 |
| 开发分支 | `embodied-tool`；不要直接修改同事的 `main` |
| 同事远程 main | `3e95a606fb987c5be57bd422a86965b44a0ecb63` |
| 工具迁移提交 | `ed50da13fd77205e53b08744ea66f0741c2a59fe`，直接基于上述 main |
| 原始工具来源 | linxiao 的 `RPent-JoyBrain`；没有重新设计训练 prompt/parser |

本交接文档、诊断入口和验证产物随 `embodied-tool` 分支一起发布，不合入 `main`。新节点以 `git rev-parse HEAD` 和 `git ls-remote origin refs/heads/embodied-tool` 核对拉取版本；不要误拉 main 后再重新搬运旧适配器代码。

Yaoxu 本地 RPent 是 `cf1247d`，也是与远程的共同祖先；远程多出同卡/分卡 GPU 调度、Joy-Brain 超时透传、resident worker 三个提交。此次真实 clone 新远程后迁移，不是拿旧目录冒充远程。“GitHub 同名重建”不等于 Git 历史被清空。

| 原适配器组件 | 当前维护位置 |
|---|---|
| `embodied_definitions/` | `robots/robotwin/embodied/definitions/` |
| `embodied_tasks.py` / `embodied_tools.py` | `robots/robotwin/embodied/tasks.py` / `tools.py` |
| `embodied_client.py` / `embodied_server.py` | `robots/robotwin/embodied/client.py` / `server.py` |
| `embodied_probe.py` / `embodied_suite.py` | `robots/robotwin/embodied/probe.py` / `suite.py` |
| 原生注册、planner 附加提示 | `robots/robotwin/embodied/plugin.py` |
| CLI、生命周期、调用计数与恢复配置 | `robots/robotwin/tool_plugin.py` 及原生 evaluate/resident/robot_spec 接入 |
| 回归测试 | `tests/unit_tests/robots/robotwin/embodied/`、`test_tool_plugin.py` |

八个工具是 `grounding_2d`、`affordance_2d`、`pointing_2d`、`trace_2d`、`trace_3d`、`grounding_3d`、`video_grounding_3d`、`depth_estimation`。八个定义及公共解析器去掉 import 后的 AST 与原实现一致，schema 由迁入的契约测试约束。

这是功能迁移，不是合并旧适配器分支的全部历史。旧 `compat_robotwin`、基线启动器、报表分析脚本和历史 50 任务大产物没有整体复制；原生 RPent 接管执行、成功判定和新实验计数。旧目录在原机器保留存档，新机器不需要其源码。

## 2. 验证进度与证据边界

已完成远程 clone、基线核对、工具迁移、普通/分卡/resume/resident 参数接入和首轮验证：

| 层级 | 已验证 | 不代表 |
|---|---|---|
| 最终全量单测 | 1446 passed / 4 failed / 1 skipped，65.45 秒；干净远程复现相同四个失败 | 所有环境组合均无失败 |
| 真实 VLM | 八工具、depth 三模式共 10/10；video 56 帧 | 任意输入准确或永不发生格式错误 |
| planner 回包结构 | 十例转换为 planner content blocks 后精确一致 | 真实 planner 已自主使用全部八种工具 |
| 原生动作接口 | 2 个原子动作步、50 个 VLA 步、4 次推理；无外部适配器导入 | 完成任务；该有限步测试官方成功为 false |
| 八卡调度 | 代码及离线 GPU 分组/worker 测试 | 已在八张 5090 上运行 |

历史证据见 [迁移记录](robots/robotwin/embodied/MIGRATION_20261014.md)。本轮两任务两卡真实 planner 对照已经完成：

| 配置 | 任务 / seed | 官方成功 / loop 验证 | VLA / 原子调用 | 执行动作步 | 感知调用 | agent loop 秒 / worker 墙钟秒 |
|---|---|---|---|---|---|---|
| 基线 | click_bell / 100000 | true / true | 2 / 1 | 117 | 0 | 354.46 / 421.92 |
| 基线 | place_container_plate / 100000 | true / true | 3 / 0 | 115 | 0 | 373.61 / 441.63 |
| 启用八工具 | click_bell / 100000 | true / true | 2 / 0 | 57 | 0 | 156.44 / 215.96 |
| 启用八工具 | place_container_plate / 100000 | true / true | 3 / 0 | 128 | 0 | 289.27 / 349.47 |

两个配置均为每卡 1 worker、共 2 worker，任务确实分配到不同 GPU 并重叠执行；基线有两个 VLA server，工具版另有 GPU 1 上的共享 VLM。所有任务的 native trace 验证通过。

**本轮自主 planner 的感知调用为零。** 因而这组结果只证明工具配置可共存并且原生闭环未被破坏，不证明 planner 在这两个简单任务中需要或使用了 VLM。不能把较短耗时归功于感知工具：VLA 随机性、动作路径和缓存均会影响结果。`agent loop` 含工具执行/等待，不是纯 API 推理耗时；worker 墙钟还包括初始化。

为补齐调用证据，另运行 [显式接口诊断](tests/e2e_tests/robotwin/embodied_planner_smoke.py)：仅此测试进程追加“先 grounding_2d、按同一步/视角 sample_world_xyz、再执行原生动作”的提示，保持原工具 schema 和官方任务目标。它不是自主 benchmark，结果独立记录，不混入上述四行。

该诊断已通过：真实 planner + 真实 VLM + 原生 Joy-Brain + 仿真，`click_bell / seed 100000`，VLA/环境在 GPU 0、VLM 在 GPU 1，共 1 worker。`grounding_2d` 和 `pointing_2d` 各成功调用一次，无感知错误；VLA 2 次、`move_to` 1 次，共 117 动作步，官方成功与 loop 验证均为 true，agent loop 392.52 秒。

完整调用顺序：`grounding_2d → sample_world_xyz → lingbot_act ×2 → pointing_2d → sample_world_xyz → move_to → finish`。grounding 返回归一化框 `[220,458,365,633]`，封装为像素框 `[70.4,109.92,116.8,151.92]` 和 `center_row_col=[130,93]`；planner 在 step 0 的 head 图上查询了这个像素。两次 VLA 后，pointing 返回 `point_xy_1000=[295,520]`、`point_row_col=[124,94]`，planner 又在 step 2 的同一视角查询该像素，再执行纠偏。

已逐项核验：两个模型产物去掉 `raw_text` 后，与真实 planner 转录中的工具返回 JSON 完全一致；两个后续 `sample_world_xyz` 调用都包含返回的像素，且 step/view 一致。不是仅靠打印日志推断 planner 收到了结果。

版本化证据：[两卡自主对照](robots/robotwin/embodied/validation/20261014/two_gpu_comparison.json)、[真实工具回包及后续调用](robots/robotwin/embodied/validation/20261014/registered_tool_diagnostic.json)。原机大日志分别保存在 `code/RPent_migration_20261013/two_gpu_handoff_20261014/`、`registered_tool_diagnostic_20261014/`；新节点不需要这些绝对路径就能阅读提交内的精简证据。

尚未验证：两 worker 同时进行感知推理的峰值/吞吐、全部八种工具在真实机器人 episode 中的使用、八卡 5090 实机运行、完整 50 任务新对照。自主工具版只加载了 VLM，真正的感知推理来自单 worker 显式诊断，不能将二者拼成已经测过并发感知性能的结论。

## 3. 后续修改必须保留的接口

- VLA 工具仍叫 `lingbot_act`，`--vla-backend joybrain` 选择原生 Joy-Brain；原子动作、qpos14 执行及官方成功判定不变。没有 `--tool-plugin` 时不得加载附加工具或改动基线 prompt。
- 工具定义 → `tasks.py` 汇总 → `tools.py` 读取已保存观察并封装 → `plugin.py` 安装和提供 planner 指导。新工具只在当前原生目录维护。
- 感知调用只读、不推进环境。`bbox_xyxy_1000` 是模型预测；像素框、`center_row_col` 由封装层根据原图尺寸派生。原文保存在产物中，planner 收到解析结果与来源，不直接收到 `raw_text`。
- 二维定位须通过同一步、同视角的 `sample_world_xyz` 等已有接口关联到世界坐标。3D 输出是相机坐标，不可直接给世界坐标的 `move_to`；预测不是碰撞检查或成功信号。
- VLM server 是 Transformers/PyTorch + WebSocket，不是 vLLM。同一个模型执行八种 prompt，推理有锁并串行处理，不因 worker 增加而自动 batching。
- 每 episode 新建感知客户端，结束只关闭自己的连接，不能停掉共享 VLM。保留源码冻结、模型/协议/prompt 身份校验及 resume 配置检查。
- resident 是实验性单任务入口，不支持常驻场景自由切换全部任务。主要接续入口是 `robots.robotwin.evaluate`。

## 4. 新节点准备

假设一个节点的作业获分配八张 GPU。`--gpu-ids` 是本节点设备列表，不是跨节点调度器；多节点集群先由平台分配单节点任务。核对实际 GPU ID 和 `CUDA_VISIBLE_DEVICES`，不得越过作业配额使用其他 GPU。

```bash
git clone --branch embodied-tool git@github.com:joy-robotics/RPent.git
cd RPent
git status --short --branch
git log -3 --oneline
git remote -v
```

使用新节点自己的 GitHub 身份。本机 `.git/config` 中的 SSH 代理辅助脚本不随 Git 提交发布，不是代码依赖。

优先复用同事在 5090 上已验证的仿真和 VLA 环境，按 [Joy-Brain 指南](robots/robotwin/JOYBRAIN.md) 和 [RoboTwin 安装指南](docs/source-zh/rst_source/simulators/robotwin.rst) 核验 RLinf、RoboTwin/SAPIEN、cuRobo、Vulkan、Joy-Brain-vla 等依赖。Python 范围由 `pyproject.toml` 定义，目前为 3.10–3.12。

配置新节点实际路径，不能照抄 linxiao/yaoxu 的路径：

```bash
export RPENT_PYTHON=/path/to/agent-simulator-env/bin/python
export VLA_PYTHON=/path/to/joybrain-policy-env/bin/python
export VLM_PYTHON=/path/to/qwen-vlm-env/bin/python
export JOYBRAIN_REPO=/path/to/Joy-Brain-vla
export JOYBRAIN_MODEL_PATH=/path/to/joybrain/pretrained_model
export ROBOTWIN_ASSETS_PATH=/path/to/RoboTwin
export VLM_CHECKPOINT=/path/to/our-qwen35-checkpoint
export RPENT_MODEL=openai:your-provider-model-id
export PLANNER_BASE_URL=https://your-approved-provider/v1
export PYTHONPATH="$PWD"
```

`OPENAI_API_KEY` 通过平台 secret 或已授权环境变量注入，不写入 Git、命令行参数、实验配置或文档。示例采用 OpenAI provider 协议，不保证模型实际由 OpenAI 托管。

本次权重身份是 Joy-Brain `mot-robotwin-stage1_100k_bs16_60k/checkpoints/060000/pretrained_model`（`robotwin_aloha`、qpos14）以及自有 VLM `qwen35_9b_midv5_sftv7_joybrain_rl_082120_sftv7_1M_v3_nothinking_n8_globalbatch1280_minibatch640_40gpu`。在新节点挂载对应权重并显式传路径，不依赖本机默认路径；不将权重或虚拟环境提交到 Git。

Agent 环境按需安装 `.[test,joybrain,embodied-tools]`；VLM 环境安装本仓库和 `.[embodied-vlm]`；VLA 环境另需 Joy-Brain-vla 的模型依赖。先确认 Torch/CUDA，不能直接让依赖解析覆盖同事已验证的环境。

本机观测版本：Python 3.11.15、Torch 2.7.1、Transformers 5.2.0、WebSockets 16.0、SAPIEN 3.0.0b1、mplib 0.2.1。它们不是 5090 环境锁文件；不要复制 H200 venv 或编译缓存。每个 GPU 环境都应检查设备、编译目标和实际运算：

```bash
nvidia-smi
"$VLM_PYTHON" - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
tensor = torch.ones((256, 256), device="cuda", dtype=torch.bfloat16)
print((tensor @ tensor).mean().item())
torch.cuda.synchronize()
PY
"$RPENT_PYTHON" -m robots.robotwin.evaluate --help
"$VLM_PYTHON" -m robots.robotwin.embodied.server --help
"$RPENT_PYTHON" -m rpent.cli.check_llm --planner api --model "$RPENT_MODEL" --base-url "$PLANNER_BASE_URL"
```

之后还须真实模型/仿真测试，不能仅凭 Torch 版本相同就认定 CUDA 构建、自定义算子、attention 和 renderer 相同。本次没有为 5090 指定未经硬件验证的二进制版本组合。

## 5. 两卡有限预算对照

本机布局为每卡 VLA server + 1 worker，共 2 worker；带工具时 GPU 1 另有共享 VLM。H200 显存容纳这种布局，不应原样套用到 5090。

先不启动 VLM，运行基线；两阶段保留相同任务、官方 seed、策略及预算：

```bash
RUN_ROOT="$PWD/logs/handoff_$(date +%Y%m%d_%H%M%S)"
COMMON=(
  --tasks click_bell,place_container_plate --episodes-per-task 1 --seed 100000
  --task-config demo_randomized --gpu-ids 0,1 --gpu-layout colocated
  --workers-per-gpu 1 --max-active-planners 2 --vla-backend joybrain
  --vla-python "$VLA_PYTHON" --vla-model-path "$JOYBRAIN_MODEL_PATH"
  --joybrain-repo "$JOYBRAIN_REPO" --robotwin-assets-path "$ROBOTWIN_ASSETS_PATH"
  --model "$RPENT_MODEL" --base-url "$PLANNER_BASE_URL" --strategy upstream
  --max-turns 12 --max-tokens 8192 --max-episode-steps 512
  --planner-timeout-s 600 --worker-startup-timeout-s 600 --joybrain-timeout 180
  --runtime-cache-dir "$RUN_ROOT/runtime_cache"
)
"$RPENT_PYTHON" -m robots.robotwin.evaluate "${COMMON[@]}" --output-dir "$RUN_ROOT/baseline"
```

只有已明确授权的明文 HTTP provider 才额外添加 `--allow-insecure-http`，不能为绕过提示擅自降低安全。访问失败先解决 planner 预检，不临时换模型掩盖问题。

基线退出并清理自有模型后，在另一终端启动共享 VLM：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLM_PYTHON" -m robots.robotwin.embodied.server --checkpoint "$VLM_CHECKPOINT" --port 19194
```

等服务就绪，在保存 `COMMON` 的终端运行：

```bash
"$RPENT_PYTHON" -m robots.robotwin.evaluate "${COMMON[@]}" \
  --tool-plugin robots.robotwin.embodied.plugin \
  --tool-plugin-endpoint ws://127.0.0.1:19194 \
  --tool-plugin-checkpoint "$VLM_CHECKPOINT" \
  --output-dir "$RUN_ROOT/embodied"
```

服务只绑定 loopback，面向同节点 worker，不将无鉴权服务暴露到外网。完成后停止自己启动的 VLM；evaluator 清理自己的 VLA/worker。

原生 evaluator 给每个 episode 独立的 `--memory-profile local --memory-dir <episode>/memory`，MemoryManager 在评测模式只读。这不是旧适配器的 `--no-memory --no-reference`，也不等于加载了历史经验；两阶段须保持同一设置。

如果自主 planner 没有调用感知工具，不要把工具计数为零记成 VLM 已验证，也不要暗中改 benchmark prompt。可单独运行显式诊断，仍使用真实 planner、VLM 和仿真，且另存输出：

```bash
"$RPENT_PYTHON" tests/e2e_tests/robotwin/embodied_planner_smoke.py \
  --task-name click_bell --task-config demo_randomized --seed 100000 \
  --vla-backend joybrain --vla-python "$VLA_PYTHON" \
  --vla-model-path "$JOYBRAIN_MODEL_PATH" --joybrain-repo "$JOYBRAIN_REPO" \
  --robotwin-assets-path "$ROBOTWIN_ASSETS_PATH" --cuda-device 0 \
  --planner api --model "$RPENT_MODEL" --base-url "$PLANNER_BASE_URL" \
  --memory-profile local --memory-dir "$RUN_ROOT/diagnostic/memory" \
  --record-actions --max-turns 16 --max-episode-steps 256 --planner-timeout-s 600 \
  --tool-plugin-endpoint ws://127.0.0.1:19194 --tool-plugin-checkpoint "$VLM_CHECKPOINT" \
  --output-dir "$RUN_ROOT/diagnostic"
```

该命令是 H200 同卡 Env/VLA 的诊断示例；5090 分卡时将 `--cuda-device 0` 改为明确的 `--vla-cuda-device 0 --env-cuda-device 1`，VLM 放在预留 GPU 7。不要和占用同一卡的 benchmark 同时启动。测试入口默认安装内置插件，提示仅在该进程生效并在退出时恢复，不改变正常 evaluator。

## 6. 在 5090 上逐步放大

先读 [原生并行指南](robots/robotwin/PARALLEL_ROLLOUT.md)。同卡默认启动门槛为 `30000 + 8000 × workers_per_gpu` MiB；分卡为 `max(30000, 8000 × workers_per_gpu)` MiB；默认运行期余量是 12000 MiB。小显存卡必须测峰值后显式配置，保护不能关闭。

同事指南的 `--gpu-start-min-free-mib 30052 --gpu-runtime-min-free-mib 2048` 是特定校准示例，不保证任意任务/checkpoint/worker 数都适用，额外加载 VLM 后须重新测量。

初次带工具验证保留独占 VLM 卡，不把两个大模型和仿真直接塞进一张卡：

| 阶段 | 配置与部署 | 并发 |
|---|---|---|
| 首个 5090 验证 | `--gpu-layout split --gpu-ids 0,1 --max-active-planners 1`；0 放 VLA，1 放环境，VLM 单独放 GPU 7 | 环境卡 1 worker |
| 扩大并发 | `--gpu-layout split --gpu-ids 0,1,2,3,4,5 --max-active-planners 3`；三组模型/环境，VLM 仍在 7，GPU 6 暂留 | `workers-per-gpu=1` 时共 3 worker |
| 纯基线参考 | 同事的八卡 colocated 或四组 split | 按显存、CPU/RAM 和实测峰值决定 |

`workers-per-gpu` 按环境卡计数，八卡 split 只有四组。以上专用 VLM 布局是待验证部署方案，不是本机已测得的 5090 吞吐。先做两任务 smoke，再加任务或 worker；VLM 串行排队变长时应区分排队、推理、planner API 和环境启动，不能盲目扩大 timeout。

## 7. 结果检查与后续任务

检查 `summary.json`、`task_summary.csv`、每个 episode 的 `evaluation.json`、`result.json`、`native_status.json` 和 `trace/`。感知还有 `tool_plugin/events.jsonl`、`metadata.json`、`embodied_model_metadata.json` 及感知图像/JSON。

真实 planner 交互见 episode 内的 `transcript_robotwin_<task>_<config>_s<seed>.json`；检查工具结果是否确实进入模型上下文，不以“已注册 schema”代替真实调用证据。本文运行的是推理/仿真评测，不是模型训练；平台作业入口应执行这些原生评测命令。

分别报告官方成功、loop 验证、动作步数、VLA/原子动作调用、每种感知工具的调用与错误、agent loop 用时和 worker 墙钟。`returncode=0` 不代表 episode 成功，解析通过不代表定位准确。

已知边界：两个 caplog 断言和两个共享 LeRobot `add_frame(task=...)` 不兼容在干净上游亦失败；affordance 空 label 严格拒绝、推理错误后客户端关闭是保留的原行为，不能编造 label 掩盖错误。修复需要真实失败样本和回归测试。

普通 evaluator 目前只接受 `demo_randomized`；两任务单 seed smoke 不是 50 任务 benchmark，也不能产出 Clean/Random/Average 三组指标。SAPIEN/cuRobo/Vulkan、5090 CUDA 扩展和八卡峰值必须在新节点验证。

Code agent 按顺序执行：确认分支 → 核查环境/设备/权重 → planner 预检 → 无工具 smoke → 带工具 smoke → 对照产物 → 扩大任务和并发。任何模型、prompt、memory、预算或 seed 变更都作为新配置记录，不覆盖旧输出。

```bash
"$RPENT_PYTHON" -m pytest tests/unit_tests/robots/robotwin -q
"$RPENT_PYTHON" -m pytest tests/unit_tests -q
```

工具开发细节见 [工具说明](robots/robotwin/embodied/README.zh-CN.md)。遵守 AGENTS 和相关 skills，不要重新迁移旧适配器。
