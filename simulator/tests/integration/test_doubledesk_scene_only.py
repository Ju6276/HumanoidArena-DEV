"""Launch the DoubleDesk evaluation scene without a policy server."""

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--hold-seconds", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.kit_args = "--/renderer/multiGpu/enabled=False --/renderer/activeGpu=0 --/physics/cudaDevice=0"
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym

import tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from tasks.common_env_config import apply_env_config_yaml  # noqa: E402


task_name = "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody"
config_path = "tasks/common_test_config/base_test/doubledesk_sonic_test.yaml"
env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=1)
apply_env_config_yaml(env_cfg, config_path, task_name=task_name, route_name="sonic")
env = gym.make(task_name, cfg=env_cfg).unwrapped
print("SCENE_ONLY_READY", flush=True)
time.sleep(args.hold_seconds)
env.close()
simulation_app.close()
