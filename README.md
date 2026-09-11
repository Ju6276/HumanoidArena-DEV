# Foundation-Model Humanoid Agent Benchmark

本仓库评估通用 Agent、VLA、WAM 作为人形机器人的高层 Brain，并比较不同低层 Body Foundation Model。核心原则是：**Brain 决定目标运动，Body 决定关节如何实现。**

```text
Brain × Body × HumanoidArena task × generalization mode × seed
```

仿真在 Brain 推理期间暂停；每次只执行经过验证的动作片段。任务成功、跌倒和 timeout 全部使用 HumanoidArena 原始 evaluator。每个 episode 连续录制 ego 和 G1 第三人称两个 50 Hz 视频，不需要人工按键或接管。

## Repository layout

```text
fm_humanoid_bench/         可安装的 Benchmark 核心包
  protocols/               reference40、native command、Brain/Body contract
  brains/                  Agent 与 HTTP VLA/WAM transport
  environments/            HumanoidArena task、mode、evaluator 路由
  evaluation/              suite、metrics、artifact、qualification
  prompts/                 reference/native 两个正式 Agent prompt
  configs/                 official、qualification、reproduction
  references/              版本化示范索引
  cli/                     episode、suite、qualification 入口
  workers/                 自主 Agent worker
simulator/                 HumanoidArena Isaac Lab 环境后端
tests/                     协议和编排测试
models/                    本地模型及可提交 manifest
external/                  固定版本的第三方 Body 源码
lerobot/                   现有 VLA 服务依赖，后续转为 external 依赖
eval_results/              本地结果，不进入源码提交
```

`simulator/` 原名 `isaaclab_twist2_g1/`。这个旧名字来自最初的 TWIST2 G1 仿真工程，已经不再用于 Benchmark 命名；TWIST2 现在位于 `external/TWIST2/`，只是五个 Body 之一。

## Unified Brain interface

| Family | 类型 | Transport | 输出 |
|---|---|---|---|
| GPT-6 | Agent | JSON stdin/stdout 或文件 transport | reference40 |
| PSI0 | VLA | HA `/reset`、`/infer` HTTP | reference40 |
| GR00T | VLA | HA `/reset`、`/infer` HTTP | reference40 |
| pi0.5 | VLA | HA `/reset`、`/infer` HTTP | reference40 |
| VLA-JEPA | VLA | HA `/reset`、`/infer` HTTP | reference40 |
| DiT4DiT | WAM | HA `/reset`、`/infer` HTTP | reference40 |

HTTP Brain 输入是 ego RGB、测量得到的 `state64` 和任务指令。Agent 每次 replanning 输入当前 ego 单帧、`task_id`、任务指令、任务描述、compact proprioception、Body contract、剩余预算和短历史；它不读取持续录制的 MP4。世界相机、reward、对象真值及 evaluator 状态不会提供给 Brain。运行使用 headless 模式，不需要人工查看视频；ego/G1 视频只在后台连续写盘供评估审计。

Agent 每次决策都会先读取对应 track 的固定模板：`prompts/reference_v31.txt` 或 `prompts/native_body_v1.txt`。runner 将模板 ID、模板 SHA-256 和完整实际请求写入 episode 的 `debug/agent/agent_*.request.json`，Agent 返回必须绑定该请求的 SHA-256；因此可以确认每一步究竟使用了哪个 prompt。配置 `--agent-command-json` 时请求通过 stdin 自动发送给 Agent 进程；文件 transport 是供外部 Agent 编排器使用的协议，本身不会伪装成一次模型调用。

`gpt6_agent` 已配置 `fm_humanoid_bench/workers/codex_agent.py`：每次 replanning 通过非交互 Codex CLI 调用 `gpt-6-astra`，附加当前 ego 图像，并用对应 track 的 JSON Schema 约束结果。无需人工创建 response 文件；模型调用失败会保存 stderr 并归类为 infrastructure error。

适配完成表示协议和执行路径存在，不表示本地一定有兼容权重。缺失或 Git LFS 未下载的权重标为 `blocked`，不进入成功率分母。当前 DiT4DiT/VLA-JEPA 本地权重与 HA 的 `state64/action40` 不匹配，因此保留适配但不伪造结果。

## Unified Body interface

| Body | reference track | native track |
|---|---|---|
| SONIC | reference40 → joint29 encoder/decoder | joint29/orientation condition |
| TWIST2 | reference40 → mimic35 | mimic35 |
| ScaleBFM | reference40 → 14-link pose mode | selected full-pose mode |
| HoloMotion | reference40 → motion actor observation | selected motion-tracking mode |
| BFM-Zero | reference40 → backward encoder → z256 | released goal/reward latent |

`reference` track 用于 Brain 与 Body 的公平互换。`native` track 允许 Brain 使用每个 Body 原生支持的控制条件，以保留不同 BFM 的能力。两个 track 共用物理步进、相机、evaluator、timeout、命名、录制和审计。接口不会为 HoloMotion 或 BFM-Zero 添加额外 `Kp × 位置误差` 伺服器。

SONIC 的 Benchmark adapter 位于 `simulator/benchmark_runtime/body_backends/legacy.py`，兼容层位于 `simulator/action_provider/`。真实权重统一安装在仓库内的 `models/body/sonic/`：两个必需 ONNX、配置文件以及两个可选 TensorRT engine；来源、大小、用途和 SHA-256 固定在同目录的 `manifest.json`。默认评估加载 ONNX Runtime CUDA，不会静默替换为 TensorRT。可用 `SONIC_POLICY_ROOT` 或 `SONIC_POLICY_DIR` 显式覆盖。

五点 keypoint 接口已经删除，不属于当前 Benchmark。

## reference40 protocol

统一动作 schema 为 `unitree_g1_gmt_refpose_v3_1`，50 Hz，每帧 40 个物理量：

| Slice | 语义 | 单位/约定 |
|---|---|---|
| `0:2` | reference-base-local XY 位移 | m/frame |
| `2` | root 高度 | m |
| `3:9` | 固定 episode reference frame 中的 root rotation6D | row-major 前两列 |
| `9:38` | canonical G1 29 个关节参考角 | rad，绝对角度 |
| `38:40` | 左/右手命令 | `<0.5` open，`≥0.5` close |

Identity rotation6D 是 `[1,0,0,1,0,0]`。Brain 输出的是身体参考轨迹，不是 motor target、joint torque 或 BFM latent。Body adapter 将参考轨迹转换成模型原生 observation/condition，BFM 最终输出 29 个关节位置 target；两者分别记录。

`state64` 是初始 heading canonical 的 root rotation6D、测量 q29 和 dq29。每个预测 chunk 在执行前完整检查 schema、形状、有限值、horizon 和能力要求。未执行的预测可被下一次 replanning 替换，已经执行的 reference anchor 和 Body history 不会回滚。

## Tasks and evaluation

保留 HumanoidArena 原有 7 个任务：OpenDoor、SitSofa、Boxing、Football、P&PBox、DoubleDesk、Vision navigation。每个任务都有 Base、Visual、Semantic、Execution 四种 mode。

| Task | 原始 timeout |
|---|---:|
| OpenDoor | 1800 tick / 36 s |
| SitSofa | 2000 tick / 40 s |
| Boxing | 900 tick / 18 s |
| Football | 2000 tick / 40 s |
| P&PBox | 1450 tick / 29 s |
| DoubleDesk | 2000 tick / 40 s |
| Vision navigation | 1800 tick / 36 s |

suite 使用 HA 原始 `sha256(task_name|group_seed|repeat_idx)` episode seed，并在所有 Brain/Body 间配对。统计区分 `success`、`fall`、`timeout`、`blocked`、`pending` 和 `infrastructure_error`，报告 coverage 和 Wilson 95% 区间。

### Frozen SONIC40 evaluation contract

旧 VLA 基线约定保留为 reproduction track：使用已有的每任务 100k checkpoint，action horizon 30；每个 checkpoint 在每个 mode 上评估 `3 seeds × 20 repeats = 60 episodes`。七任务、四 mode 的一个完整 Brain family 因此是 `7 × 4 × 60 = 1,680 episodes`。输入固定为一张 `[480,640,3] uint8` ego RGB、`state64` 和规范英文任务指令，输出固定为 `[30,40] float32` semantic reference trajectory。模型内部 normalization 可以不同，仿真收到的物理量和左右手 `0/1` 语义必须一致。

`fm_humanoid_bench/configs/reproduction/psi0_sonic40_table_s7.json` 精确表达现有 PSI0→SONIC Table S7 的 1,680-episode population。训练数据来源通过 model catalog 的 `training_body` 保存；SONIC 与 TWIST2 数据训练出的 checkpoint 不会被当成同一个 Brain 条件。

`fm_humanoid_bench/configs/official/reference.json` 是跨 Body transfer inventory sweep，不是固定 benchmark 样本数。当前本地 catalog 会生成 38,400 个**计划** job：任务专用 checkpoint 按其支持任务展开到五个 Body，task-agnostic GPT-6 Agent 则覆盖七任务。随着 Brain/checkpoint 增减，这个数量会改变，因此 38,400 不能写成 benchmark 的固有规模。

这些 suite 文件只定义评估 episode，不启动或统计训练任务。`training_body`、checkpoint step 和训练数据来源仅作为已训练权重的 provenance。

## Demonstration conditions

- `none`：不提供示范。
- `state_action`：提供选中 episode 的 state64/action40 采样。
- `images`：提供选中 episode 的 ego 关键帧。

reference catalog 按任务、modality、bundle hash 和 payload hash 固定。Agent 可在第一个物理动作前从公开 ID 菜单选择一次，之后该 episode 不再改变。数值条件不泄露示范图片，图片条件不泄露示范状态/动作。目前只登记了一条 OpenDoor episode，包含 8 个状态/动作采样和 8 张 V-JEPA2.1 选择的 ego 图。

## Output contract

每个 suite 根目录只保留一个 `README.md`，并用 `summary.json`/`task_metrics.csv` 汇总 benchmark cell、用 `episodes.csv`/`episodes.jsonl` 索引每条 episode。`videos/success/` 和 `videos/failure/<reason>/` 提供按结果浏览的相对软链接，不复制视频。每个正式 episode 使用一个目录：

```text
manifest.json
metrics.json
task_metrics.csv
task_evaluation.json
evaluation_protocol.json
provenance.json
recording_verification.json
interface_audit.json
brain_decisions.jsonl
simulation_timeline.jsonl
episode.npz
videos/ego.mp4
videos/g1.mp4
debug/                    推理帧、Agent 请求/响应和 runner log
```

成功与失败 episode 都保留 `episode.npz` 和两路完整视频，才能复算指标并检查失败模式。用于论文展示的精选视频可以另建索引，但不能替代完整原始结果。推理等待不会产生额外 simulation tick 或视频帧。基础设施失败不会改写成任务失败；replay、controller probe 和 inference smoke 也不能记作 closed-loop success。

## Commands

从仓库根目录运行：

```bash
# 仅生成不可变计划，不启动模型或仿真
python fm_humanoid_bench/cli/run_suite.py fm_humanoid_bench/configs/official/reference.json \
  --root eval_results/foundation_reference

# 复现旧 PSI0→SONIC Table S7：7 tasks × 4 modes × 60 episodes
python fm_humanoid_bench/cli/run_suite.py fm_humanoid_bench/configs/reproduction/psi0_sonic40_table_s7.json \
  --root eval_results/psi0_sonic40_table_s7_reproduction

# 执行/恢复一个限制数量的子集
python fm_humanoid_bench/cli/run_suite.py fm_humanoid_bench/configs/qualification/integration.json \
  --root eval_results/integration --run --max-jobs 2

# 运行单个 episode；Brain server 地址按实际模型设置
python fm_humanoid_bench/cli/run_episode.py \
  --brain psi0 --body sonic --task open_door --mode Base \
  --server http://127.0.0.1:18080

# 查看 reference catalog
python fm_humanoid_bench/cli/references.py list \
  --catalog fm_humanoid_bench/references/catalog.json \
  --task open_door --condition images

# CPU contract/orchestration tests
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests -p 'test_*.py' -q
```

## Current validation

- 74 项协议、Agent prompt transport、七任务描述、产物索引与编排测试通过。
- SONIC、TWIST2 使用真实 ONNX 权重通过 provider 对照和 74 个输入扰动检查。
- 五个 Body 均完成 600 帧 reference40 物理控制探针和逐帧审计；SONIC、TWIST2、ScaleBFM、HoloMotion 通过资格门槛，BFM-Zero 因主动关节跟踪 MAE `0.1688 rad` 超过 `0.15 rad` 门槛而未通过。
- PSI0 的正式历史任务结果保留；PSI0、GR00T 的 HTTP `30×40` 协议、启动配置和失败分类已接入统一 runner。
- 同一 runner、同一 SitSofa seed 下，PSI0→SONIC 和 PSI0→TWIST2 都通过原始 HA 成功判定，连续双视角录制和逐帧重算通过。
- 同一条成功的 GPT-6 OpenDoor reference40 决策流已在五个 Body 上完成固定回放：ScaleBFM 和 BFM-Zero 触发 HA 原始成功条件，SONIC、TWIST2、HoloMotion 到原始 36 秒 timeout；所有 episode 的 ego/G1 视频连续性和逐帧接口重算均通过。汇总位于 `eval_results/body_open_door_regression_20260911/summary.json`。
- 自主 Codex worker 已配置 `gpt-6-astra`；reference/native prompt 装载、ego 图像传输、JSON Schema 和请求 SHA-256 绑定由协议测试覆盖。临时 smoke 产物不作为正式结果保留。

这些结果验证接口闭环，不代表 38,400 个计划 episode 已经跑完。

## License and citation

本仓库集成 HumanoidArena、TWIST2、SONIC/GR00T Whole-Body Control、LeRobot、ScaleBFM、HoloMotion 和 BFM-Zero。发布或再分发前应分别检查源码、模型和资产许可。

```bibtex
@article{wang2026humanoidarena,
  title={HumanoidArena: Benchmarking Egocentric Hierarchical Whole-body Learning},
  author={Wang, Taowen and Xie, Zikang and Yang, Bin and others},
  journal={arXiv preprint arXiv:2606.17833},
  year={2026}
}
```
