RoboCasa GR1
===========

使用 Agent 选择 Joy-Brain 动作和关节目标修正，运行原生 GR1 桌面任务。
入口为 ``--robot robocasa_gr1``，与 RoboCasa365/PandaOmron、RoboTwin 后端独立。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      原始 Joy-Brain VLA，使用 ``robocasa_gr1`` 统计量。

   .. grid-item-card:: 规划器

      RPent 工具调用规划器，包括 API 规划器。

   .. grid-item-card:: 任务

      原生 GR1 桌面任务的 ID 和 OOD 划分。

   .. grid-item-card:: 硬件

      CUDA 模型推理与 MuJoCo EGL 渲染；并发前先实测显存。

任务
~~~~

.. list-table:: 原始 Joy-Brain GR1 任务列表
   :header-rows: 1

   * - ``--eval-type``
     - 范围
   * - ``id``
     - 24 个分布内任务。
   * - ``ood_object_appearance``
     - 18 个物体外观泛化任务。
   * - ``ood_container_combination``
     - 14 个容器组合泛化任务。
   * - ``ood_object_type``
     - 32 个物体类别泛化任务。

任务归属读取指定 Joy-Brain 源码中的 ``benchmark/Robocasa_GR1/task_lists.py``，
不使用 RoboCasa365 的 Target50 列表。

观测与动作
~~~~~~~~~~

.. list-table:: 原生接口
   :header-rows: 1

   * - 接口
     - 行为
   * - 观测
     - 原生第一视角 RGB、任务语言、关节和回合状态。VLA 接收 raw44 状态与一张图像；裁剪和归一化由原始服务端负责。
   * - 动作
     - 29 维有效关节目标，依次为左臂7、左手6、右臂7、右手6、腰部3。raw44 保留槽位遵循训练映射，不是任意补零适配。
   * - 奖励与成功判定
     - 与原始 GR1 客户端一致，累计原生 ``info.success`` 或原生终止信号。预算截断和规划器文字不算成功。
   * - 任务指令
     - 以原生观测中的任务语言为准。

安装与资源准备
--------------

分别配置 RPent Agent、原始 Joy-Brain 模型、原生 GR1 仿真三个 Python 环境。
只在 RPent 专用环境中安装轻量客户端依赖：

.. code-block:: bash

   JD_SOURCE=https://mirrors.jd.com/pypi/simple
   /path/to/rpent/python -m pip install -e '.[robocasa-gr1]' --index-url "$JD_SOURCE"

复用已有 ``robocasa-gr1-tabletop-tasks`` 分支及资产。原生运行时要求 MuJoCo 3.2.6、
robosuite 1.5.0/1.5.1，并需使用其支持的 NumPy 版本。不要把该分支装进 PandaOmron 环境。
指定完整 GR1 checkpoint，包含 ``config.json``、``stats.json``、``model.safetensors``，
并核实配置引用的 VLM/processor 文件。后端不会自动选择 checkpoint。

.. code-block:: bash

   export JOYBRAIN_REPO_ROOT=/path/to/Joy-Brain-vla
   export JOYBRAIN_MODEL_PATH=/path/to/explicit/gr1/checkpoint/pretrained_model
   export JOYBRAIN_PYTHON=/path/to/model-env/bin/python
   export ROBOCASA_GR1_REPO="$JOYBRAIN_REPO_ROOT/third_party/robocasa-gr1-tabletop-tasks"
   export ROBOCASA_GR1_PYTHON=/path/to/gr1-env/bin/python
   export ROBOCASA_GR1_LD_LIBRARY_PATH=/path/to/gr1-env/lib:/path/to/verified/gl-libraries

在目标节点核实 GL 库目录。GPU 序号以获分配节点为准；EGL 默认采用 Env 的单个数字 CUDA
设备序号，UUID 或多设备掩码需显式指定 ``--egl-device-id``。
仅在获分配的计算节点运行，不在登录节点启动 GPU 负载。

运行一个任务
------------

先使用新的输出目录执行有限组件验收：

.. code-block:: bash

   /path/to/rpent/python -m tests.e2e_tests.robocasa_gr1.smoke \
     --task-name PnPCupToDrawerClose --seed 7 --cuda-device 0 \
     --max-episode-steps 17 --output-dir logs/gr1-components

该入口依次验证原生 reset/RGB、仅推理、原生关节保持和一个 VLA 动作窗口，
**不调用 Agent API**。它是独立入口，不属于 ``run_gpu_suite.sh`` 的可选目标。

确认 API 凭据和仿真图像发送权限后，通过正式 Agent 入口执行：

.. code-block:: bash

   /path/to/rpent/python -m rpent.cli.main --robot robocasa_gr1 \
     --planner api --model openai:YOUR_MODEL --base-url "$AGENT_BASE_URL" \
     --memory-profile local --task-name PnPCupToDrawerClose --seed 7 \
     --cuda-device 0 --max-episode-steps 32 --replan-steps 16 \
     --max-turns 6 --max-tokens 2048 --output-dir logs/gr1-agent

通过环境变量提供凭据，不放进命令参数或受版本控制的文件。
规划器配置见 :doc:`../guides/configure_planner`。
``gr1_act(chunks=1..4)`` 获取完整预测，每个窗口执行 ``--replan-steps`` 个动作，
且不超过剩余原生动作预算。默认值为16，与原始启动脚本一致，不修改模型 chunk。
``set_joint_targets`` 仅提供原生关节分组控制，未指定的分组保持当前关节位置。

查看结果
--------

``result.json`` 将原生成败与规划器 finish 分开记录。
执行不确定或记录失败时，``measurement_valid=false``，``official_success`` 保持未知，
不强行给出任务成败结论。
``states.json``、第一视角图片和 ``episode.mp4`` 记录观测轨迹；
``trace/step_*/transition.json`` 记录真实执行动作与原生判定。
``proposal_*.npy`` 是完整预测，不代表全部执行。
服务端日志与 ``policy-metadata.json`` 用于追溯运行时和模型。
组件验收另记录分阶段耗时、设备级采样峰值显存。
组件通过不代表任务成功、Agent 吞吐量或基准成绩。

任务记忆
--------

仅支持 ``--memory-profile local``，不下载或复用 PandaOmron 记忆。
此后端尚未实现探索模式及多回合常驻 worker；每个环境只拥有一个记录回合，不能重复 reset。

故障排查
--------

复用服务端时配置 ``--vla-endpoint ws://HOST:PORT`` 和同一显式 checkpoint 路径。
客户端会验证 GR1 元信息，且不会停止外部服务端。
分卡时使用 ``--env-cuda-device`` 与 ``--vla-cuda-device``，不要同时传 ``--cuda-device``。
这不等于已实现并行布局启动器，也不证明同卡显存容量。

资产或 EGL 问题查看 ``env-server.log``，VLM 文件或 CUDA 问题查看 ``vla-server.log``。
状态或动作维度不符会报错，不通过裁维度适配。
动作执行不确定或记录失败会阻止后续动作；应新建验收作业，不重放不确定动作。
