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

CPU 回归：`python -m pytest -q tests/unit_tests/robots/robotwin`。
真实组件测试入口为 [joybrain_smoke.py](../../tests/e2e_tests/robotwin/joybrain_smoke.py)，
使用同样的 VLA/环境参数，另传 `--output`；该测试不调用高层 LLM，不代表任务成功率。

并行评测入口为 `python -m robots.robotwin.evaluate`，数据导出入口为
`python -m robots.robotwin.dataset`；用法见 [PARALLEL_ROLLOUT.md](PARALLEL_ROLLOUT.md)。
