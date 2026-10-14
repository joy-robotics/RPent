# RoboTwin 的 Joy-Brain VLA 后端

通过 `--vla-backend joybrain` 选择后端，默认仍为 LingBot。客户端、启动器和
动作适配、记录器和评测入口都位于本仓库。

## 依赖与启动

先按 [RoboTwin 指南](../../docs/source-zh/rst_source/simulators/robotwin.rst)
配置 RPent、RLinf 0.4、RoboTwin 和 cuRobo，再安装客户端可选依赖：

```bash
python -m pip install -e ".[joybrain]"
```

本地启动模型还需要 Joy-Brain-vla 源码、匹配 `robotwin_aloha` 的 checkpoint
及模型 Python 环境；仿真需要 RoboTwin 资产。将示例路径和模型 ID 替换为实际
配置，在 RPent 根目录运行：

```bash
python -m rpent.cli.main --robot robotwin \
  --vla-backend joybrain \
  --joybrain-repo /path/to/Joy-Brain-vla \
  --vla-model-path /path/to/pretrained_model \
  --vla-python /path/to/policy-env/bin/python \
  --robotwin-assets-path /path/to/RoboTwin \
  --task-name click_bell --task-config demo_clean --seed 0 \
  --memory-profile local --output-dir /path/to/new/run \
  --planner api --model 'openai:your-model-id'
```

规划 API 凭据使用标准环境变量，不写入源码。加 `--record-actions` 可保存所有
原子工具和 VLA 的真实动作及对齐观测；默认使用上游 prompt，
`--strategy atomic-preferred` 可显式选择偏原子搬运/放置的策略。

- `--vla-model-path` 可由 `JOYBRAIN_MODEL_PATH` 提供；它不与 LingBot 权重混用。
- `--joybrain-repo` 默认取 `JOYBRAIN_REPO_ROOT`，否则查找同级 Joy-Brain-vla 目录。
- `--vla-python` 默认当前 Python；建议模型与 RPent 分环境运行。
- 已有服务可用 `--vla-endpoint ws://host:port` 连接，不在本地加载模型。
  此时可额外传 `--vla-model-path` 核对服务声明的 checkpoint 身份。
- `--joybrain-timeout` 默认 120 秒；GPU 分配沿用 `--env-cuda-device` 和 `--vla-cuda-device`。

## 行为边界

保留上游规划、原子技能和官方成功判定。工具仍叫 `lingbot_act`，忽略局部
`prompt` 参数，使用环境任务指令。Joy-Brain 接收三路 RGB 和 qpos14 状态，
输出 qpos14，经原生关节目标接口执行；LingBot 仍使用 EEF16/EE 接口。

Joy-Brain 通常用 `16+16+16+2` 填充 50 动作窗口。预处理和反归一化由模型后端
负责，客户端仅按约定二值化夹爪。控制接口和推理频率有差异，不能称为严格只换权重。
协议、形状和有限值校验失败不会自动重放；执行结果不确定时禁止继续执行动作。

## 验证与并行

常驻 worker 的独立试验入口为 `python -m robots.robotwin.resident_evaluate`。
当前只支持同卡每卡一份 VLA、一个常驻 worker 和单一任务；会保留进程，但按原生
流程刷新每个 seed 的场景/语言并隔离会话与记录器。见 [RESIDENT_ROLLOUT.md](RESIDENT_ROLLOUT.md)。
下述默认 `evaluate` 入口仍是每个 episode 新建 worker，未被常驻模式替换。

CPU 回归：`python -m pytest -q tests/unit_tests/robots/robotwin`。
真实组件测试入口为 [joybrain_smoke.py](../../tests/e2e_tests/robotwin/joybrain_smoke.py)，
使用同样的 VLA/环境参数，另传 `--output`；该测试不调用高层 LLM，不代表任务成功率。

并行评测入口为 `python -m robots.robotwin.evaluate`，数据导出入口为
`python -m robots.robotwin.dataset`；用法见 [PARALLEL_ROLLOUT.md](PARALLEL_ROLLOUT.md)。

正式评测入口支持两种布局，`--workers-per-gpu` 始终表示每张**环境卡**上的
worker 数。GPU 列表可以使用实际索引或完整 UUID，必须全部位于当前获分配的
可见设备范围内，不会自动替换缺失设备。

| 布局 | GPU 参数 | 模型数 / worker 槽位数 |
| --- | --- | --- |
| 八卡同卡，各一个 worker | `--gpu-layout colocated --gpu-ids 0,1,2,3,4,5,6,7 --workers-per-gpu 1` | 8 / 8 |
| 四组分卡，共 20 worker | `--gpu-layout split --gpu-ids 0,1,2,3,4,5,6,7 --workers-per-gpu 5` | 4 / 20 |

分卡模式按列表顺序组成 `(模型卡, 环境卡)`，上例即 `0→1、2→3、4→5、6→7`。
每组只有模型卡加载 VLA，环境卡上的五个 worker 共用该组 endpoint。
模型内部推理仍为串行，没有自动 batch。

先用明确的小任务预算生成计划，例如：

```bash
python -m robots.robotwin.evaluate \
  --gpu-layout split --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers-per-gpu 5 --max-active-planners 20 \
  --joybrain-repo "$JOYBRAIN_REPO_ROOT" \
  --vla-model-path "$JOYBRAIN_MODEL_PATH" --vla-python "$JOYBRAIN_PYTHON" \
  --robotwin-assets-path "$ROBOTWIN_ASSETS_PATH" \
  --model 'openai:your-model-id' --tasks click_bell --episodes-per-task 1 \
  --gpu-start-min-free-mib 30052 --gpu-runtime-min-free-mib 2048 \
  --max-turns 4 --planner-timeout-s 600 --max-episode-steps 256 \
  --dry-run --output-dir /path/to/new/plan
```

这些环境变量必须先设置为真实路径，模型 ID 和 API 凭据需匹配获准使用的服务商。
示例只规划一个 episode，不会因为有 20 个槽位就自动启动 20 条任务。
30052/2048 MiB 是经过特定配置校准的示例值，不是所有权重、任务或 GPU 的通用保证。
确认实际显存、任务预算和仿真图像/上下文的发送授权后，才移除 `--dry-run`。

同卡模式的默认启动门槛为 `30000 + 8000 × workers` MiB；分卡模式在每张所选卡上
默认要求 `max(30000, 8000 × workers)` MiB。运行期默认保留 12000 MiB，可显式配置，
但必须满足 `启动门槛 >= 运行门槛 >= 1024 MiB`。两种检查覆盖所有模型卡和环境卡。

每张环境卡一次只初始化一个 worker，等其初始 `states.json` 观测落盘后再放行下一个；
已就绪的 worker 可以继续执行，不同卡也可以同时初始化。初始化等待默认 1800 秒，
可用 `--worker-startup-timeout-s` 设置；超时记录为该次尝试失败，不自动重放。
控制器为所有模型先启动进程，再等待就绪，并通过不执行推理的保活请求防止慢启动时
模型服务空闲退出。

默认在输出目录的 `runtime-cache` 下按模型 GPU、worker 槽位和组件隔离编译/渲染缓存，
可用 `--runtime-cache-dir` 设置根目录。同一槽位在先前进程组退出后才可复用。
不会写入既有模型环境或修改系统驱动。

**这里复用的是 VLA 服务和 worker 槽位，不是跨 episode 常驻的 Python worker/仿真环境。**
每个 episode 仍创建新的原生 CLI worker。跨 episode 常驻复用及长期吞吐需要另行验证，
不能把 dry-run 或工程动作链通过当作完整 Agent 成功。

`plan.json` 记录模型/环境分组、模型服务数和总 worker 容量；每条调度结果区分
`gpu_uuid`（环境卡）与 `model_gpu_uuid`。恢复任务默认继承原布局，老结果按同卡布局
处理；显式更改布局、设备数量或 worker 数会保留各阶段来源，并标为混合并发配置。
显存门槛需重新显式提供，不能把旧布局的余量静默套用到新配置。
