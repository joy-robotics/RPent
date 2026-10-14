# RPent 原生工具迁移与上游核验（2026-10-14）

本文记录首次迁移时的证据；后续两卡闭环验证、远程发布和 5090 接续进度见 [当前交接文档](../../../HANDOFF_EMBODIED_TOOLS.md)。

## “同名重建”的含义

同事的原话描述的是 GitHub 上的仓库被重新创建，不能据此推断 Git 提交历史被清空、代码从零重写，或本地旧 checkout 已经自动更新。是否兼容应以实际远程引用、共同祖先和实现为准。仅凭代码也不能独立证明 GitHub 上曾发生删除操作。

本次通过已有 GitHub SSH 身份从 `joy-robotics/RPent` 全量 clone，并在 2026-10-14 再次查询远程引用，结果如下：

| 对象 | 提交 |
|---|---|
| 远程 `main` / 新 RPent 的基线 | `3e95a606fb987c5be57bd422a86965b44a0ecb63` |
| `/mnt/workspace/users/yaoxu/RPent` 的本地 `main` | `cf1247d2faf1323f5af90d587815a94920ea8212` |
| 两者共同祖先 | `cf1247d2faf1323f5af90d587815a94920ea8212` |

`git rev-list --left-right --count cf1247d...main` 返回 `0 3`：远程保留了本地旧历史，并多出三个提交，不是两条互不相关的历史。

| 提交 | 日期（UTC+8） | 实际变更 |
|---|---|---|
| `8c93b0b` | 2026-10-12 | 同卡部署与模型/环境分卡部署的并行测评、资源检查及缓存 |
| `efc3e3a` | 2026-10-13 | 将 Joy-Brain 请求超时参数传到 worker |
| `3e95a60` | 2026-10-13 | 单任务、跨 episode 的常驻 Agent/Env worker |

因此，yaoxu 这个本地目录并不完全等于当前远程。此次以真正 clone 的远程为基线迁移，没有拿旧目录冒充远程，也没有覆盖同事的目录。

## 原生接口与多卡能力

- `robot_spec.py` 通过 `--vla-backend joybrain` 选择仓库内的 `JoyBrainClient`；planner 仍调用原有 `lingbot_act` schema。后端接收真实多视角图像和 qpos14 状态，返回 qpos14 动作。
- 原子动作继续由原生 toolkit/primitives 执行；官方任务成功状态继续由 RoboTwin 提供。基线不需要导入 `rpent_joybrain`。
- 普通 `evaluate` 支持 `colocated`（每张卡上的模型与环境共置）及 `split`（按顺序两卡配对）。例如八卡 split 是 `0→1, 2→3, 4→5, 6→7`，不是自动等价于八个 VLA worker。
- `workers-per-gpu` 按环境卡计数。默认启动显存门槛分别为 `30000 + 8000 × workers` MiB 和 `max(30000, 8000 × workers)` MiB。约 32GB 显存卡不能机械套用同卡默认值；需按原生 `PARALLEL_ROLLOUT.md` 的实测配置设定门槛，保护检查不会因使用小卡而自动取消。
- 新增的 resident 入口仍限一个 `demo_randomized` 任务，每卡一个常驻 worker，要求每卡至少两个 episode；并不是已经支持 50 个任务在同一常驻场景中无缝切换。
- 这里只在本地 H200 做了真实接口验证，未声称实测八卡 5090 的吞吐或 50 任务成功率。

## Embodied tools 的统一维护位置

八个工具全部在 `robots/robotwin/embodied/definitions/` 定义，经 `tasks.py` 汇总，在 `tools.py` 封装，通过 `plugin.py` 接入原生 toolkit：

`grounding_2d`、`affordance_2d`、`pointing_2d`、`trace_2d`、`trace_3d`、`grounding_3d`、`video_grounding_3d`、`depth_estimation`。

保持原有 prompt、SPEC 和解析器语义；八个定义文件及公共解析器去除 import 后的 AST 与旧适配器一致。归一化坐标到像素、row/column 的派生结果继续由工具封装层产生。模型原文保存在实验产物中，不直接塞进 planner 工具回包。

基线不传 `--tool-plugin`；带工具运行显式指定 `--tool-plugin robots.robotwin.embodied.plugin`、共享 VLM endpoint 和 checkpoint，无须指定另一个仓库的根目录。普通、split、resume、resident 路径都沿用原生入口；插件源码冻结、每 episode 客户端关闭及独立调用计数已接入。各 worker 共享借用的 VLM 服务，关闭客户端不停止共享服务。

VLM 服务仍是 Transformers/PyTorch + WebSocket，不是 vLLM。此次保留 affordance 空 label 严格校验和推理错误后客户端关闭的原行为；十例通过不代表任意输入永远不会失败。

## 验证证据

下列 JSON 是实际运行产物，不是模拟成功结果，保存在同目录的 `validation/20261014/`：

| 检查 | 结果 | 边界 |
|---|---|---|
| 真实本地 VLM、八工具及 depth 三模式 | 10/10 通过；video 使用 56 帧 | 使用训练参考数据，不衡量未知场景准确率 |
| 工具回包传入 planner content blocks | 十例序列化往返精确一致 | 未调用真实外部 planner API |
| 原生 Joy-Brain + 原子动作 | GPU 0，`place_container_plate`，seed `100000`，约 157.56 秒 | 有限步接口测试，不是任务成功 |
| 上述动作执行 | 原子动作 2 步；VLA 50 步、4 次推理；无外部适配器导入 | 官方 `eval_success=false`，不隐瞒或改写 |
| 最终 `code/RPent` 路径全量单测 | 1442 passed / 4 failed / 1 skipped，64.05 秒 | Python 3.11.15，借用已有环境；与移动前结果一致 |
| 干净远程 main 同环境全量单测 | 1196 passed / 4 failed / 1 skipped | 四个失败名称一致，不是迁移新增失败 |
| 格式与静态检查 | 本次检查文件格式通过；新增代码 lint 通过 | 四个上游文件已有 CPY001，未补版权头 |

四个已复现的单测失败：

- dashboard 关闭异常的 `caplog` 捕获断言。
- session 清理失败警告的 `caplog` 捕获断言。
- 两个 Flywheel 导出测试：当前共享 LeRobot 的 `add_frame()` 不接受 `task=` 参数。

没有为获得绿灯而改变上游日志、导出代码或共享依赖。环境未安装 `pre-commit`，因此未声称执行完整 hooks；本次文档为 Markdown，未改 Sphinx 页面。全量测试需要宿主机本地端口；沙箱内中断的那次不作为有效通过率证据。

## 保存与复现

详细日志及大文件保留在 `/mnt/workspace/users/linxiao/code/RPent_migration_20261013/`，包括 `remote_full_unit_tests.log`、`remote_clean_unit_tests_host.log`、`final_path_unit_tests.log`、`native_vlm_remote_20261014/` 和 `native_joybrain_remote_20261014/`。目录名为 13 日是因为迁移从当日开始，真实验证日期见产物中的 14 日时间戳。

原 `code/RPent` 整体保留为迁移目录内的 `original_RPent/`，其未提交修改、`.git` 历史及实验文件不丢弃；`code/RPent-JoyBrain` 和 yaoxu 路径不修改。开发入口是新的 `code/RPent` 的 `embodied-tool` 分支；首次迁移仅做本地提交，后续发布情况以当前交接文档为准。

服务启动、原生测评和探针的参数示例见 [工具说明](README.zh-CN.md)。真实 planner/50 任务对照实验及八卡 5090 硬件运行不属于这次已完成的接口验证，不能从上述烟测结果推导。
