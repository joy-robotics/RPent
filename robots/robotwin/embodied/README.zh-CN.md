# 只读 Embodied Tools

[English](README.md)

[2026-10-14 迁移与验证报告](MIGRATION_20261014.md)

本包在 RPent 原生 RoboTwin toolkit 中按需增加八种 VLM 感知工具，不再依赖另一个适配器仓库。
不设置 `--tool-plugin` 时，不加载工具和附加 planner 提示；LingBot、Joy-Brain、原子动作、
官方成功判定、memory 设置及测评预算均保持原有行为。

## 代码位置

| 文件 | 职责 |
|---|---|
| `definitions/*.py` | 每个工具的 `PROMPT`、`SPEC`、parser 和 `TASK` 注册 |
| `tasks.py` | 输入校验、解析分发、prompt 指纹 |
| `server.py` | Transformers/PyTorch Qwen3.5 模型的 WebSocket 服务 |
| `client.py` | checkpoint/协议/prompt 握手与推理请求 |
| `tools.py` | 保存的观察、坐标派生结果与实验产物 |
| `plugin.py` | 原生 toolkit 注册、planner 指导、源码指纹 |
| `probe.py` | 保存图像上的推理与外部 planner 完整回传核验 |
| `suite.py` | 八工具及 depth 三种模式的参考数据烟测 |

`../tool_plugin.py` 管理可选插件生命周期、CLI 参数及独立调用计数。
原生 `robots.robotwin.evaluate` 负责向 worker 传递同一配置、冻结源码、
在已有任务报表中增加每种感知工具的调用次数，不替换 VLA 或原子动作。
NumPy 传输直接复用 `rpent.utils.msgpack_numpy`。

## 工具与返回

| 工具 | 结果集合 | 含义 |
|---|---|---|
| `grounding_2d` | `detections` | 物体框，含 0..1000 归一化及原图像素坐标 |
| `affordance_2d` | `regions` | 交互区域框，不保证是可行抓取位姿 |
| `pointing_2d` | `points` | 归一化/像素点及整数 `[row,column]` 索引 |
| `trace_2d` | `trajectories` | 图像平面中的有序末端执行器关键点 |
| `trace_3d` | `trajectories` | 被操作物体的相机坐标轨迹，单位米 |
| `grounding_3d` | `detections` | 相机坐标物体框、尺寸和旋转编码 |
| `video_grounding_3d` | `detections` | 相对于最后一帧相机的物体框 |
| `depth_estimation` | `depths` | 相机到物体/标记/指定点的单个距离，不是深度图 |

所有工具只读保存的观察，不推进环境。向 `sample_world_xyz` 传入像素索引时，必须使用返回的
同一 `step_idx/view`。框中心是确定性算出的几何位置，不是第二次模型预测，也不保证是接触点。
3D 预测不会自动变换到机器人世界坐标。

`success=true` 表示推理及解析完成，`found` 表示预测集合非空，均不代表官方任务成功。
Planner 收到结构化数据及来源信息，不直接收到 `raw_text`。
`<episode>/embodied_tools/<request_id>.json` 另外保留模型原文，并保存输入图和 overlay。
包括失败在内的调用记录位于 `<episode>/tool_plugin/events.jsonl`。

## 环境和服务

原生 worker 使用现有 RoboTwin 环境；轻量 `embodied-tools` extra 提供 Pillow、WebSockets、MsgPack。
模型服务使用独立 CUDA 环境，`embodied-vlm` extra 增加 Transformers 5.2+ 和 PyTorch，
不要混入版本固定的 LingBot 环境。具体 PyTorch/CUDA 构建须匹配实际 GPU；安装 extra 不等于
验证 RTX 5090 驱动和运行环境。

以下命令均在仓库根目录执行；`VLM_PYTHON` 指向模型环境，`VLM_CHECKPOINT` 指向 Qwen3.5 多模态权重：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$VLM_PYTHON" -m robots.robotwin.embodied.server \
  --checkpoint "$VLM_CHECKPOINT" --port 19194
```

这不是 vLLM，也不是 OpenAI-compatible HTTP 接口。后端使用 bf16、SDPA、greedy、关闭 thinking，
默认最多生成 1024 token；一个模型服务八种 prompt，加锁串行推理。无认证服务仅绑定本机 loopback。
服务由操作者独立启动；worker 结束只关闭连接，不停止共享服务。Agent 进程不需要导入 Torch。
不传 checkpoint 时仍使用移植前的本地默认路径；换机器时应显式传入。客户端会核对路径、协议和 prompt 指纹。

## 原生测评

仍以 `--vla-backend joybrain` 选择原生 Joy-Brain 后端。
按现有 Joy-Brain 文档配置 VLA 源码/权重、RoboTwin assets、policy Python 和 provider 凭据。
双卡、每卡一个 worker 的示例：

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

内置插件不需要 `--tool-plugin-root`。无感知工具基线删除三个 `--tool-plugin*` 参数，并不启动 VLM。
对照实验须保持 task/seed、strategy、memory、预算一致；原生 evaluator 的 memory 默认配置不等同于
历史 `--no-memory --no-reference` 实验，必须明确选择。明文 HTTP planner 地址还需要原生的
`--allow-insecure-http` 显式确认。

调度器接受物理 GPU 标识，八卡可用 `--gpu-ids 0,1,2,3,4,5,6,7`；这是配置示例，不是八卡实测声明。
共享 VLM 与同卡 VLA/环境的额外资源占用需单独预留。插件不会改变原生动作步数统计和官方成功判定。

标准 evaluator 的启动空闲显存默认值，在 `--gpu-layout colocated` 下为每卡
`30000 + 8000 * workers_per_gpu` MiB，在 `--gpu-layout split` 下为每卡
`max(30000, 8000 * workers_per_gpu)` MiB；split 使用模型卡与环境卡配对。
设置 `--gpu-start-min-free-mib` 和 `--gpu-runtime-min-free-mib` 时须依据实测预留安全余量，
两项保护均保留。插件不修改这些原有控制；填写八个 GPU ID 不等于验证了具体的 5090 配置。

同样的三个插件参数也适用于 `robots.robotwin.resident_evaluate`。该实验性入口一次只支持一个
任务、每卡一个常驻 worker，且每卡至少两个 episode；它不是跨任务复用场景的入口。
每个 episode 建立新的感知客户端和观测状态，独立启动的 VLM 服务继续共享。
源码清单包含插件的 prompt/parser，汇总中的 `perception_tool_calls` 统计调用总数，
各 episode 仍保留 `perception_tool_usage`。原生进程复用、场景重置、memory 隔离和资源保护不变。

## 独立验证

```bash
"$RPENT_PYTHON" -m robots.robotwin.embodied.probe detect \
  --image "$OBSERVATION_PNG" --tool grounding_2d --target 'the red cup' \
  --checkpoint "$VLM_CHECKPOINT" --output "$NEW_PROBE_DIR"

"$RPENT_PYTHON" -m robots.robotwin.embodied.suite \
  --checkpoint "$VLM_CHECKPOINT" --data-root "$GROUNDING_DATA" \
  --spatial-data-root "$SPATIAL_DATA" --output "$NEW_SUITE_DIR"
```

Suite 默认覆盖八种工具，depth 包含 object/marker/point 三种情况；`--audit-only` 只检查参考模板，
不调用模型。3D probe 需要保存的相机 metadata，video probe 需要按顺序追加图像。

将 `detect` 改为 `planner` 并提供 `--model/--base-url`，可核验真实 planner 完整接收。
密钥来自 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 或 `--api-key-env` 指定环境变量。
可选 `--planner-config` 读取 JSON 的 `planner` 配置（model/base URL 或其环境变量名），
但不再自动读取其他应用的私有凭据文件。Suite 的 `--planner` 接受相同参数。
该模式会把测试任务和工具输出发送到所选外部 provider；保存图像的 probe 不等于实时机器人测评。

CPU 离线测试：

```bash
python -m pytest tests/unit_tests/robots/robotwin -q
```

## 保留的限制

本次迁移保留原有 prompt 和预测契约：affordance 空标签仍严格拒绝，客户端也仍保留推理错误后关闭
连接的行为。没有把这些已知问题悄悄改成成功；修复它们需要单独验证的行为变更。
输出格式有效也不保证几何正确、无碰撞或任务成功。
