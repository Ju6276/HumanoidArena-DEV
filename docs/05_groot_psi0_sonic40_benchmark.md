# GR00T N1.7 / Psi0 / VLA-JEPA / DiT4DiT on HumanoidArena: complete SONIC40 workflow

This guide describes the exact five-repository setup used to train and evaluate
GR00T N1.7, Psi0, the official VLA-JEPA baseline, and DiT4DiT on the seven
HumanoidArena SONIC tasks. HumanoidArena owns the simulator and evaluation
protocol; each model repository owns its model, training environment,
checkpoint, and inference server.

The comparison in this guide uses one policy per task, 100,000 optimizer steps,
action horizon 30, global batch size 8, W&B online, and 60 base-test evaluation
episodes per checkpoint (three seeds and 20 repeats per seed).

## 1. Repositories and responsibilities

Keep the five repositories next to each other:

```text
/path/to/benchmark/
├── HumanoidArena/       simulator, SONIC controller integration, evaluation
├── Isaac-GR00T/         GR00T N1.7 data adapter, training, inference bridge
├── Psi0/                Psi0 training and HumanoidArena inference server
├── VLA-JEPA/            official VLA-JEPA training and inference server
├── DiT4DiT/             official DiT4DiT training and inference server
├── data_v2/             shared converted LeRobot v2.1 datasets
└── releases/            downloaded archives, models, and simulator assets
```

Use the matching development branches:

| Repository | Branch | Role |
| --- | --- | --- |
| `Ju6276/HumanoidArena-DEV` | `feat/humanoidarena-vla-adapters` | Evaluation and SONIC `semantic_v3` execution |
| `Ju6276/Isaac-GR00T-DEV` | `feat/humanoidarena-sonic40` | GR00T N1.7 training and serving |
| `Ju6276/Psi0-DEV` | `feat/humanoidarena-sonic40` | Psi0 training and serving |
| `Ju6276/VLA-JEPA-DEV` | `feat/humanoidarena-sonic40` | Official VLA-JEPA training and serving |
| `Ju6276/DiT4DiT-DEV` | `feat/humanoidarena-sonic40` | Official DiT4DiT training and serving |

```bash
export BENCH_ROOT=/path/to/benchmark
mkdir -p "${BENCH_ROOT}"
cd "${BENCH_ROOT}"

git clone --branch feat/humanoidarena-vla-adapters \
  https://github.com/Ju6276/HumanoidArena-DEV.git HumanoidArena
git clone --recurse-submodules --branch feat/humanoidarena-sonic40 \
  https://github.com/Ju6276/Isaac-GR00T-DEV.git Isaac-GR00T
git clone --branch feat/humanoidarena-sonic40 \
  https://github.com/Ju6276/Psi0-DEV.git Psi0
git clone --branch feat/humanoidarena-sonic40 \
  https://github.com/Ju6276/VLA-JEPA-DEV.git VLA-JEPA
git clone --branch feat/humanoidarena-sonic40 \
  https://github.com/Ju6276/DiT4DiT-DEV.git DiT4DiT
```

For reproducible runs, record the five commit hashes:

```bash
git -C "${BENCH_ROOT}/HumanoidArena" rev-parse HEAD
git -C "${BENCH_ROOT}/Isaac-GR00T" rev-parse HEAD
git -C "${BENCH_ROOT}/Psi0" rev-parse HEAD
git -C "${BENCH_ROOT}/VLA-JEPA" rev-parse HEAD
git -C "${BENCH_ROOT}/DiT4DiT" rev-parse HEAD
```

The VLA-JEPA SONIC40 adapter starts directly from `ginwind/VLA-JEPA` main at
`ec8c70f`; its adapter commit is `72a6ccb`. It deliberately does not inherit the
separate V-JEPA 2.1, latent78, music-JEPA, or one-stage experimental branches.

## 2. Frozen benchmark interface

All models receive and produce the same values. This is the interface that
HumanoidArena expects during evaluation.

### Input

- One front RGB image: `[480,640,3]`, HWC, `uint8`.
- State: `[64]`, `float32`.
- One canonical English task instruction.

State order:

```text
[0:6]    heading-canonical root rotation, row-layout rotation 6D
[6:35]   canonical Unitree G1 joint positions, 29D
[35:64]  canonical Unitree G1 joint velocities, 29D
```

### Output

- Action chunk: `[30,40]`, `float32`.
- Action format: `semantic_v3`.

Action order at every horizon step:

```text
[0:2]    reference-root base-local XY delta
[2:3]    reference-root Z
[3:9]    reference-root row-layout rotation 6D
[9:38]   canonical Unitree G1 joint reference positions, 29D
[38:40]  left and right binary hand commands
```

The 40D output is a semantic reference-pose command. It is not a language-model
token and is not SONIC's native 64D encoder latent. HumanoidArena interprets the
40D command and sends the corresponding reference to the SONIC whole-body
controller. Always use:

```bash
export SONIC_VLA_ACTION_FORMAT=semantic_v3
```

Do not use `latent64` for this comparison.

### Internal normalization versus the frozen interface

The raw dataset fields and simulator-facing values are identical, but each
model retains its own training-time normalization:

| Model | Continuous action `[0:38]` | Binary hands `[38:40]` | Simulator-facing hands |
| --- | --- | --- | --- |
| GR00T N1.7 | q01/q99 to `[-1,1]` | q01/q99 (`0/1` becomes `-1/+1`) | denormalized to `0/1` |
| Psi0 | min/max to `[-1,1]` | min/max (`0/1` becomes `-1/+1`) | denormalized to `0/1` |
| VLA-JEPA | min/max to `[-1,1]` | binary identity in `[0,1]` | thresholded to `0/1` |
| DiT4DiT | min/max to `[-1,1]` | binary identity in `[0,1]` | thresholded to `0/1` |
| official pi0.5 | q01/q99 to `[-1,1]` | q01/q99 (`0/1` becomes `-1/+1`) | denormalized to `0/1` |

This is an internal algorithm/preprocessor difference, not a benchmark
interface difference. Every server must return the same raw `[30,40]`
`semantic_v3` action chunk, and HumanoidArena thresholds dimensions 38 and 39
as the left/right hand commands. Do not feed normalized model-space values
directly to the simulator.

### Tasks and prompts

| Short name | HumanoidArena task ID | Prompt |
| --- | --- | --- |
| `opendoor` | `HSI_open_door` | `Open the door.` |
| `double_desk` | `HOI_double_desk` | `Put the hammer from the right table into the basket on the left table.` |
| `football` | `HOI_football` | `Kick the soccer ball into the goal.` |
| `pp_box` | `HOI_pp_box` | `Move the box from the table onto the shelf.` |
| `boxing` | `HSI_boxing` | `Strike the green markers on the punching bag.` |
| `sit_sofa` | `HSI_sit_sofa` | `Sit on the sofa.` |
| `vision_navi` | `HSI_vision_navi` | `Avoid obstacles and move to the yellow marked area.` |

## 3. Install HumanoidArena and SONIC

HumanoidArena evaluation uses a dedicated Isaac Lab environment. Follow
[`docs/04_environment_setup.md`](04_environment_setup.md) to install Isaac Sim
5.0.0, Isaac Lab `release/2.2.0`, and the project requirements.

Download and restore the simulation assets so these directories exist:

```text
HumanoidArena/isaaclab_twist2_g1/assets/objects
HumanoidArena/isaaclab_twist2_g1/assets/robots
```

Download the SONIC encoder and decoder:

```bash
cd "${BENCH_ROOT}/HumanoidArena"
mkdir -p GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
hf download nvidia/GEAR-SONIC \
  model_encoder.onnx model_decoder.onnx observation_config.yaml \
  --local-dir GR00T-WholeBodyControl/gear_sonic_deploy/policy/release

export SONIC_POLICY_ROOT="${BENCH_ROOT}/HumanoidArena/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
export ARENA_ROOT="${BENCH_ROOT}/HumanoidArena"
export EVAL_PYTHON=/path/to/unitree_sim_env/bin/python
```

Verify the simulator environment before loading a VLA model:

```bash
OMNI_KIT_ACCEPT_EULA=YES "${EVAL_PYTHON}" -c \
  "import isaacsim; print('isaacsim import ok')"
test -f "${SONIC_POLICY_ROOT}/model_encoder.onnx"
test -f "${SONIC_POLICY_ROOT}/model_decoder.onnx"
```

## 4. Download and convert the dataset once

The Git repositories do not contain the large dataset. Download the released
HumanoidArena V3.1 archive from ModelScope:

```bash
python -m pip install -U modelscope
mkdir -p "${BENCH_ROOT}/releases/HumanoidArenaV3.1"
modelscope download --dataset Twang2026/HumanoidArenaV3.1 \
  --local_dir "${BENCH_ROOT}/releases/HumanoidArenaV3.1"
```

Dataset page:
<https://www.modelscope.cn/datasets/Twang2026/HumanoidArenaV3.1>

The released archive is approximately 29 GB. If a Git clone produces a
136-byte `.zip`, that file is only a Git LFS pointer; run `git lfs pull` or use
the ModelScope CLI above before extracting it.

```bash
export DATA_ARCHIVE="${BENCH_ROOT}/releases/HumanoidArenaV3.1/HumanoidArena_datasets_v3_1.zip"
test "$(stat -c%s "${DATA_ARCHIVE}")" -gt 1000000000 || {
  echo "Dataset archive is missing or is only an LFS pointer" >&2
  exit 1
}
mkdir -p "${BENCH_ROOT}/releases/HumanoidArenaV3.1/extracted"
unzip "${DATA_ARCHIVE}" \
  -d "${BENCH_ROOT}/releases/HumanoidArenaV3.1/extracted"
find "${BENCH_ROOT}/releases/HumanoidArenaV3.1/extracted" \
  -path '*/meta/info.json' -print
```

Extract the archive, then identify the seven SONIC task directories. Each input
directory passed below must itself contain `meta/info.json`. Do not pass a
TWIST2 directory or a mixed-backend dataset.

All model adapters consume the same LeRobot V2.1 data. Convert the V3
release once with the GR00T wrapper and keep the converted datasets in a shared
directory:

Here, “V3.1” is the HumanoidArena protocol/release version; the source
LeRobot container may report `codebase_version: v3.0`. The converted container
must report `codebase_version: v2.1` while retaining HumanoidArena protocol
version 3.1 in `vla_protocol`.

```bash
export SONIC_V3_ROOT=/path/to/extracted/sonic/task_directories
export DATA_ROOT="${BENCH_ROOT}/data_v2"
mkdir -p "${DATA_ROOT}"

cd "${BENCH_ROOT}/Isaac-GR00T"
for task in opendoor double_desk football pp_box boxing sit_sofa vision_navi; do
  uv run python examples/HumanoidArena/prepare_dataset.py \
    --input "${SONIC_V3_ROOT}/${task}" \
    --output "${DATA_ROOT}/humanoidarena_sonic_v31_${task}"
done
```

If the extracted task directories have different names, replace only the
`--input` values; keep the seven output names unchanged. The wrapper:

1. copies the source so the downloaded release is not modified;
2. invokes GR00T's official `scripts/lerobot_conversion/convert_v3_to_v2.py`
   when the source is V3;
3. installs the shared modality metadata;
4. rejects non-SONIC data and validates image/state/action dimensions and order.

If an output already exists, inspect it rather than overwriting it blindly.
Use `--overwrite` only when intentionally rebuilding that exact output.

Expected final layout:

```text
data_v2/
├── humanoidarena_sonic_v31_opendoor/
├── humanoidarena_sonic_v31_double_desk/
├── humanoidarena_sonic_v31_football/
├── humanoidarena_sonic_v31_pp_box/
├── humanoidarena_sonic_v31_boxing/
├── humanoidarena_sonic_v31_sit_sofa/
└── humanoidarena_sonic_v31_vision_navi/
```

Each dataset should report LeRobot `v2.1`, `observation.state[64]`,
`action[40]`, `backend_source=sonic`, and the
`unitree_g1_gmt_refpose_v3_1` schema.

## 5. Train GR00T N1.7

### 5.1 Environment and base model

```bash
cd "${BENCH_ROOT}/Isaac-GR00T"
uv sync --python 3.10
uv run python -c "import gr00t; print('GR00T installed')"

uv run hf download nvidia/GR00T-N1.7-3B \
  --local-dir "${BENCH_ROOT}/releases/GR00T-N1.7-3B"
```

The training scripts also accept `nvidia/GR00T-N1.7-3B` directly, but a fixed
local snapshot is preferable for reproducibility.

### 5.2 Eight-GPU batch configuration

With eight A100 GPUs and global batch size 8, GR00T computes:

```text
per_device_train_batch_size = global_batch_size / num_gpus = 8 / 8 = 1
gradient_accumulation_steps = 1
effective optimizer batch = 8
```

Set both `CUDA_VISIBLE_DEVICES` and `NUM_GPUS`. Merely exposing eight GPUs does
not override the wrapper's one-GPU default.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export GLOBAL_BATCH_SIZE=8
export BASE_MODEL_PATH="${BENCH_ROOT}/releases/GR00T-N1.7-3B"
export WANDB_MODE=online
export WANDB_ENTITY=your-wandb-entity
```

### 5.3 Train OpenDoor

```bash
cd "${BENCH_ROOT}/Isaac-GR00T"
DATASET_PATH="${DATA_ROOT}/humanoidarena_sonic_v31_opendoor" \
  uv run bash examples/HumanoidArena/train_opendoor_sonic.sh
```

Defaults: 100,000 optimizer steps, horizon 30, global batch 8, checkpoint every
5,000 steps, and W&B online.

### 5.4 Train the remaining tasks separately

```bash
cd "${BENCH_ROOT}/Isaac-GR00T"
for task in double_desk football pp_box boxing sit_sofa vision_navi; do
  DATASET_PATH="${DATA_ROOT}/humanoidarena_sonic_v31_${task}" \
    uv run bash examples/HumanoidArena/train_sonic_task.sh "${task}"
done
```

This is sequential: the next process starts only after the current training
process exits. Each task produces a separate policy. Do not train a combined
seven-task policy if the goal is the per-task comparison.

The alternative unattended wrapper trains and evaluates each of the remaining
six tasks before advancing. This wrapper expects the standard dataset names
under `Isaac-GR00T/datasets`, so link the shared converted data there first:

```bash
cd "${BENCH_ROOT}/Isaac-GR00T"
mkdir -p datasets
for task in double_desk football pp_box boxing sit_sofa vision_navi; do
  ln -s "${DATA_ROOT}/humanoidarena_sonic_v31_${task}" \
    "datasets/humanoidarena_sonic_v31_${task}"
done

ARENA_ROOT="${ARENA_ROOT}" \
EVAL_PYTHON="${EVAL_PYTHON}" \
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT}" \
NUM_GPUS=8 GLOBAL_BATCH_SIZE=8 \
bash examples/HumanoidArena/train_eval_sonic_queue.sh
```

Use the manual loop instead if a person must approve each evaluation before the
next training run starts.

### 5.5 Evaluate a GR00T checkpoint

The evaluator starts `serve_eval_stack.py`, which starts the native GR00T policy
server and the HumanoidArena HTTP bridge. Point `MODEL_PATHS_CSV` at the final
GR00T checkpoint directory.

```bash
export GROOT_REPO="${BENCH_ROOT}/Isaac-GR00T"
export GROOT_CHECKPOINT=/path/to/gr00t/run/checkpoint-100000
export SERVER_PYTHON="${GROOT_REPO}/.venv/bin/python"
export SERVER_SCRIPT="${GROOT_REPO}/examples/HumanoidArena/serve_eval_stack.py"

cd "${ARENA_ROOT}"
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON="${EVAL_PYTHON}" \
SERVER_PYTHON="${SERVER_PYTHON}" \
SERVER_SCRIPT="${SERVER_SCRIPT}" \
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT}" \
MODEL_PATHS_CSV="${GROOT_CHECKPOINT}" \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml \
RESULTS_DIR="${ARENA_ROOT}/eval_results/groot_n17_opendoor_base_smoke" \
SEEDS_OVERRIDE=0 REPEATS_PER_SEED=1 \
RESUME_LATEST=0 \
NUM_WORKERS=1 SERVER_GPU_IDS=0 ISAAC_DEVICE=cuda:0 \
HEADLESS=1 RECORD_VIDEO_EVERY_N=0 \
SONIC_VLA_ACTION_FORMAT=semantic_v3 \
bash isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/HSI_open_door_run_vla_eval_parallel.sh
```

After the smoke result contains one real episode and no `process_error` or
`worker_error`, run the formal 60 episodes:

```bash
cd "${ARENA_ROOT}"
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON="${EVAL_PYTHON}" \
SERVER_PYTHON="${SERVER_PYTHON}" \
SERVER_SCRIPT="${SERVER_SCRIPT}" \
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT}" \
MODEL_PATHS_CSV="${GROOT_CHECKPOINT}" \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml \
RESULTS_DIR="${ARENA_ROOT}/eval_results/groot_n17_opendoor_base_$(date +%Y%m%d_%H%M%S)" \
SEEDS_OVERRIDE='0 1 2' REPEATS_PER_SEED=20 \
RESUME_LATEST=0 \
NUM_WORKERS=1 SERVER_GPU_IDS=0 ISAAC_DEVICE=cuda:0 \
HEADLESS=1 RECORD_VIDEO_EVERY_N=1 \
SONIC_VLA_ACTION_FORMAT=semantic_v3 \
bash isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/HSI_open_door_run_vla_eval_parallel.sh
```

## 6. Train Psi0

### 6.1 Environment

```bash
cd "${BENCH_ROOT}/Psi0"
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync \
  --group serve --group viz --group psi \
  --index-strategy unsafe-best-match --active
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
python -c "import psi; print(psi.__version__)"
```

Copy `.env.sample` to `.env` and configure at least the Hugging Face token and
W&B account:

```bash
cp .env.sample .env
# Edit HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, and PSI_HOME.
```

### 6.2 Official Psi0 initialization weights

```bash
cd "${BENCH_ROOT}/Psi0"
source .venv-psi/bin/activate

python scripts/data/download.py \
  --repo-id=USC-PSI-Lab/psi-model \
  --remote-dir=psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
  --local-dir=cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
  --repo-type=model
python scripts/data/download.py \
  --repo-id=USC-PSI-Lab/psi-model \
  --remote-dir=psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
  --local-dir=cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
  --repo-type=model
```

### 6.3 Eight-GPU batch configuration and training

Psi0 explicitly divides the requested global batch by GPU count and gradient
accumulation. With eight GPUs, global batch 8, and accumulation 1, each GPU also
receives one sample.

```bash
cd "${BENCH_ROOT}/Psi0"
source .venv-psi/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLOBAL_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export DATA_ROOT="${BENCH_ROOT}/data_v2"
export WANDB_MODE=online
export WANDB_ENTITY=your-wandb-entity

bash scripts/train/psi0/finetune-humanoidarena-sonic40-psi0.sh opendoor
```

Train the other tasks sequentially:

```bash
for task in double_desk football pp_box boxing sit_sofa vision_navi; do
  bash scripts/train/psi0/finetune-humanoidarena-sonic40-psi0.sh "${task}"
done
```

The script validates the dataset before training. Defaults are 100,000 optimizer
steps, horizon 30, global batch 8, DDP bf16, checkpoint every 5,000 steps, and
W&B online.

### 6.4 Evaluate a Psi0 checkpoint

For Psi0, `MODEL_PATHS_CSV` must point to the training run directory containing
`run_config.json`, `argv.txt`, and `checkpoints/`, not directly to
`checkpoints/ckpt_100000`.

```bash
export PSI_REPO="${BENCH_ROOT}/Psi0"
export PSI_RUN_DIR=/path/to/psi0/run-directory
export SERVER_PYTHON="${PSI_REPO}/.venv-psi/bin/python"
export SERVER_SCRIPT="${PSI_REPO}/src/psi/deploy/psi0_serve_humanoidarena.py"

cd "${ARENA_ROOT}"
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON="${EVAL_PYTHON}" \
SERVER_PYTHON="${SERVER_PYTHON}" \
SERVER_SCRIPT="${SERVER_SCRIPT}" \
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT}" \
MODEL_PATHS_CSV="${PSI_RUN_DIR}" \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml \
RESULTS_DIR="${ARENA_ROOT}/eval_results/psi0_opendoor_base_$(date +%Y%m%d_%H%M%S)" \
SEEDS_OVERRIDE='0 1 2' REPEATS_PER_SEED=20 \
RESUME_LATEST=0 \
NUM_WORKERS=1 SERVER_GPU_IDS=0 ISAAC_DEVICE=cuda:0 \
HEADLESS=1 RECORD_VIDEO_EVERY_N=1 \
SONIC_VLA_ACTION_FORMAT=semantic_v3 \
bash isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/HSI_open_door_run_vla_eval_parallel.sh
```

The OpenDoor-only all-in-one wrapper performs training, one smoke episode, and
then 60 formal episodes:

```bash
cd "${PSI_REPO}"
source .venv-psi/bin/activate
ARENA_ROOT="${ARENA_ROOT}" \
EVAL_PYTHON="${EVAL_PYTHON}" \
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT}" \
DATA_ROOT="${DATA_ROOT}" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
GLOBAL_BATCH_SIZE=8 \
bash scripts/train/psi0/run-humanoidarena-opendoor-train-eval.sh
```

Do not run this wrapper after OpenDoor has already been trained unless another
100,000-step run is intended.

## 7. Train VLA-JEPA

### 7.1 Official baseline and environment

This branch is based on the official `ginwind/VLA-JEPA` main and uses its
original V-JEPA 2 encoder, `facebook/vjepa2-vitl-fpc64-256`. It is not the
separate V-JEPA 2.1 experiment.

```bash
cd "${BENCH_ROOT}/VLA-JEPA"
conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

The base checkpoints may be referenced by Hugging Face ID or fixed local
snapshots:

```text
Qwen/Qwen3-VL-2B-Instruct
facebook/vjepa2-vitl-fpc64-256
```

### 7.2 Eight-GPU training

VLA-JEPA trains all approximately 2.77B parameters with DeepSpeed ZeRO-2 and
bf16. Use eight A100 GPUs for the frozen comparison. With global batch 8 and
gradient accumulation 1, each GPU receives one sample.

```bash
cd "${BENCH_ROOT}/VLA-JEPA"
conda activate VLA_JEPA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export GLOBAL_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export DATA_ROOT="${BENCH_ROOT}/data_v2"
export WANDB_MODE=online
export WANDB_ENTITY=your-wandb-entity

bash scripts/vlajepa_humanoidarena_sonic40.sh opendoor
```

Train the other tasks separately and sequentially:

```bash
for task in double_desk football pp_box boxing sit_sofa vision_navi; do
  bash scripts/vlajepa_humanoidarena_sonic40.sh "${task}"
done
```

Defaults are 100,000 optimizer steps, horizon 30, global batch 8, checkpoint
every 5,000 steps, W&B online, Qwen input at 224x224, and the official V-JEPA 2
world-model path with eight 256x256 frames. The external evaluator still sends
one current `[480,640,3]` front RGB frame; model-native resizing remains an
internal preprocessing detail.

### 7.3 Evaluate a VLA-JEPA checkpoint

`MODEL_PATHS_CSV` points directly to a checkpoint `.pt` file such as
`steps_100000_pytorch_model.pt`. The checkpoint's run directory must also
contain `config.yaml` and `dataset_statistics.json`.

```bash
export VLAJEPA_REPO="${BENCH_ROOT}/VLA-JEPA"
export VLAJEPA_CHECKPOINT=/path/to/vlajepa/run/checkpoints/steps_100000_pytorch_model.pt
export SERVER_PYTHON=/path/to/VLA_JEPA/bin/python
export SERVER_SCRIPT="${VLAJEPA_REPO}/examples/HumanoidArena/serve_humanoidarena.py"

cd "${ARENA_ROOT}"
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON="${EVAL_PYTHON}" \
SERVER_PYTHON="${SERVER_PYTHON}" \
SERVER_SCRIPT="${SERVER_SCRIPT}" \
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT}" \
MODEL_PATHS_CSV="${VLAJEPA_CHECKPOINT}" \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml \
RESULTS_DIR="${ARENA_ROOT}/eval_results/vlajepa_opendoor_base_smoke" \
SEEDS_OVERRIDE=0 REPEATS_PER_SEED=1 RESUME_LATEST=0 \
NUM_WORKERS=1 SERVER_GPU_IDS=0 ISAAC_DEVICE=cuda:0 \
HEADLESS=1 RECORD_VIDEO_EVERY_N=0 \
SONIC_VLA_ACTION_FORMAT=semantic_v3 \
bash isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/HSI_open_door_run_vla_eval_parallel.sh
```

After the one-episode smoke has no `process_error` or `worker_error`, use a new
results directory, `SEEDS_OVERRIDE='0 1 2'`, `REPEATS_PER_SEED=20`, and
`RECORD_VIDEO_EVERY_N=1` for the formal 60 episodes.

## 8. Train DiT4DiT

This adapter starts from the official `Mondo-Robotics/DiT4DiT` main branch and
keeps its Cosmos-Predict2.5-2B + ActionDiT architecture, AdamW optimizer, bf16,
Accelerate, and DeepSpeed ZeRO-2 training stack. Only the dataset contract,
64D/40D ordering, horizon 30 configuration, and HTTP serving boundary are
adapted.

```bash
cd "${BENCH_ROOT}/DiT4DiT"
conda create -n dit4dit python=3.10 -y
conda activate dit4dit
pip install -e .

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export GLOBAL_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=1
export DATA_ROOT="${DATA_ROOT}"
export BASE_MODEL="${BENCH_ROOT}/releases/Cosmos-Predict2.5-2B-diffusers-base-post-trained"
export OUTPUT_ROOT="${BENCH_ROOT}/checkpoints/dit4dit_humanoidarena"
export WANDB_MODE=online
export WANDB_ENTITY=your-wandb-entity
export WANDB_PROJECT=HumanoidArena

bash scripts/dit4dit_humanoidarena_sonic40.sh opendoor
for task in double_desk football pp_box boxing sit_sofa vision_navi; do
  bash scripts/dit4dit_humanoidarena_sonic40.sh "${task}"
done
```

The formal setup is 100,000 optimizer steps per task and
`global batch = 8 processes * 1 sample/GPU * 1 accumulation = 8`. It trains one
separate policy per task. The wrapper validates the shared LeRobot v2.1 data
before starting and keeps W&B online.

Point evaluation at a checkpoint `.pt` file. The run directory must contain
the matching `config.yaml` and `dataset_statistics.json`:

```bash
export DIT4DIT_REPO="${BENCH_ROOT}/DiT4DiT"
export DIT4DIT_CHECKPOINT=/path/to/dit4dit/run/final_model/pytorch_model.pt
export SERVER_PYTHON=/path/to/dit4dit/bin/python
export SERVER_SCRIPT="${DIT4DIT_REPO}/examples/HumanoidArena/serve_humanoidarena.py"

cd "${ARENA_ROOT}"
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON="${EVAL_PYTHON}" \
SERVER_PYTHON="${SERVER_PYTHON}" \
SERVER_SCRIPT="${SERVER_SCRIPT}" \
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT}" \
MODEL_PATHS_CSV="${DIT4DIT_CHECKPOINT}" \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml \
RESULTS_DIR="${ARENA_ROOT}/eval_results/dit4dit_opendoor_base_smoke" \
SEEDS_OVERRIDE=0 REPEATS_PER_SEED=1 RESUME_LATEST=0 \
NUM_WORKERS=1 SERVER_GPU_IDS=0 ISAAC_DEVICE=cuda:0 \
HEADLESS=1 RECORD_VIDEO_EVERY_N=0 \
SONIC_VLA_ACTION_FORMAT=semantic_v3 \
bash isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/HSI_open_door_run_vla_eval_parallel.sh
```

After the smoke succeeds, use a fresh results directory,
`SEEDS_OVERRIDE='0 1 2'`, `REPEATS_PER_SEED=20`, and
`RECORD_VIDEO_EVERY_N=1` for the formal 60 episodes. A single 48 GB 4090D can
load the full model and run `[1,30,40]` inference, but full-parameter ZeRO-2
optimizer initialization exceeds its memory; use the planned 8×A100 setup for
training rather than changing the formal configuration.

## 9. Evaluate the other six tasks

Use the same evaluation templates and replace the checkpoint/run directory,
config, result name, maximum episode length, and task wrapper:

| Short name | Config under `tasks/common_test_config/base_test/` | Wrapper | `MAX_STEPS` |
| --- | --- | --- | ---: |
| `opendoor` | `open_door_sonic_test.yaml` | `HSI_open_door_run_vla_eval_parallel.sh` | 1800 |
| `double_desk` | `doubledesk_sonic_test.yaml` | `HOI_double_desk_run_vla_eval_parallel.sh` | 2000 |
| `football` | `football_single_sonic_test.yaml` | `HOI_football_run_vla_eval_parallel.sh` | 2000 |
| `pp_box` | `pp_box_sonic_test.yaml` | `HOI_pp_box_run_vla_eval_parallel.sh` | 1450 |
| `boxing` | `boxing_sonic_test.yaml` | `HSI_boxing_run_vla_eval_parallel.sh` | 900 |
| `sit_sofa` | `sit_sofa_sonic_test.yaml` | `HSI_sit_sofa_run_vla_eval_parallel.sh` | 2000 |
| `vision_navi` | `vision_navi_sonic_test.yaml` | `HSI_vision_navi_run_vla_eval_parallel.sh` | 1800 |

All wrappers are under:

```text
HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/
```

For the comparison described here, retain:

```bash
SEEDS_OVERRIDE='0 1 2'
REPEATS_PER_SEED=20
SONIC_VLA_ACTION_FORMAT=semantic_v3
```

## 10. Official pi0.5 reference evaluation

Download the released HumanoidArena models from:
<https://www.modelscope.cn/models/Twang2026/HumanoidArena_models>

Keep the released `pi/<task>/...` structure, then use the same
`sonic_pi05/*_run_vla_eval_parallel.sh` task wrapper with the official pi0.5
checkpoint selected through `MODEL_PATHS_CSV` or `MODEL_ROOT`. This gives the
reference model the same SONIC backend, scene config, seeds, repeats, maximum
steps, and `semantic_v3` action interpretation as GR00T, Psi0, VLA-JEPA, and
DiT4DiT.

Do not convert a pi0.5 checkpoint into a GR00T or Psi0 checkpoint. Only the
dataset and HumanoidArena observation/action contract are shared.

## 11. Evaluation modes

This workflow fixes `ENV_CONFIG_YAML` under `base_test` for the direct model
comparison. HumanoidArena also provides three distribution-shift modes:

| Mode | Meaning |
| --- | --- |
| `base_test` | Demonstration-like object ranges, textures, and lighting |
| `semantic` | Semantic material/texture changes and task-area distractors |
| `vision` | Lighting randomization |
| `execution` | Wider task-relevant asset randomization ranges |

Do not average these four modes together unless the reporting protocol calls
for that aggregate. Report the mode next to every success rate. To evaluate
another mode, select the matching YAML under
`tasks/common_test_config/<mode>/` while keeping the model, seeds, repeats, and
task limits fixed.

## 12. Results and success-rate calculation

Every formal result directory should contain:

```text
summary.jsonl
episodes/
logs/
```

Count results and compute success rate:

```bash
"${EVAL_PYTHON}" - /path/to/results/summary.jsonl <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
successes = sum(bool(row.get("success")) for row in rows)
print(f"episodes={len(rows)} successes={successes} success_rate={successes / len(rows):.3%}")
bad = [row for row in rows if row.get("failure_reason") in {"process_error", "worker_error"}]
print(f"infrastructure_failures={len(bad)}")
PY
```

A formal run is complete only when it contains 60 distinct episode rows. Treat
`process_error` and `worker_error` as infrastructure failures to diagnose, not
ordinary policy failures.

## 13. Recommended execution order

For every algorithm and task:

1. Validate the converted dataset.
2. Train to step 100,000 and confirm the final checkpoint exists.
3. Confirm W&B reports global batch 8, horizon 30, and the correct task.
4. Run one headless smoke episode with video disabled.
5. Check that the server received `[480,640,3] + [64]` and returned `[30,40]`.
6. Run 60 base-test episodes with video enabled.
7. Confirm `summary.jsonl` has 60 rows and no infrastructure failures.
8. Save the five Git commit hashes, checkpoint path, W&B run URL, and result path.
9. Start the next task only after the previous result has been checked.

## 14. Common mistakes

- Downloading the Git LFS pointer instead of the 29 GB dataset archive.
- Training directly from V3 instead of the converted V2.1 task directory.
- Mixing SONIC and TWIST2 demonstrations.
- Using the 64D SONIC latent output instead of the canonical 40D semantic action.
- Changing the 40D order or treating the reference pose as an observation-relative residual.
- Lowercasing or otherwise changing the canonical task prompt in only one model.
- Setting `CUDA_VISIBLE_DEVICES=0,1,...,7` but leaving GR00T `NUM_GPUS=1`.
- Pointing Psi0 evaluation at `ckpt_100000` instead of its parent run directory.
- Pointing VLA-JEPA evaluation at a run directory instead of its checkpoint `.pt` file.
- Pointing DiT4DiT evaluation at a run directory instead of its checkpoint `.pt` file.
- Training the custom V-JEPA 2.1 branch when the intended baseline is official VLA-JEPA/V-JEPA 2.
- Starting the 60-episode run before a one-episode smoke test passes.
- Comparing different scene modes, seed counts, repeat counts, or maximum step limits.
