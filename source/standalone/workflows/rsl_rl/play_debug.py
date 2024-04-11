# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# local imports
import cli_args  # isort: skip
import numpy as np
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Aliengo-Z1-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import traceback

import carb
from rsl_rl.runners import OnPolicyRunner

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_onnx,
)

get_data = False

def main():
    saved_action = np.load("/home/elgceben/orbit/action.npy")
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    saved_action = torch.tensor(saved_action, device=env.device, dtype=torch.float).reshape(-1, 12)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = "/home/elgceben/orbit/logs/rsl_rl/aliengo_z1_rough/fix_arm/model_29999.pt" # relatively, 40000 is best
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(ppo_runner.alg.actor_critic, export_model_dir, filename="policy.onnx")

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    i = 0
    action_list = []
    position_list = []
    velocity_list = []
    obs_list = []
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if get_data:
                obs_list.append(obs.cpu().numpy())
                obs[:, 9:12] = torch.tensor([0.4, 0.0, 0.0], device=env.device, dtype=torch.float)
                actions = policy(obs)
                obs_list.append(obs.cpu().numpy())
            # if i >= 100:
            if not get_data:
                actions = torch.zeros((env.num_envs, 12), device=env.device, dtype=torch.float)
                
                actions[:, :] = saved_action[i, :]
                obs[:, 9:12] = torch.tensor([0.4, 0.0, 0.0], device=env.device, dtype=torch.float)
                obs_list.append(obs.cpu().numpy())
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            action_list.append(actions.cpu().numpy())
            position_list.append(obs[:, 12:31].cpu().numpy())
            velocity_list.append(obs[:, 31:50].cpu().numpy())
            
        i += 1
        if get_data:
            if i == 1200:
                np.save("action.npy", np.array(action_list))
                np.save("position.npy", np.array(position_list))
                np.save("velocity.npy", np.array(velocity_list))
                np.save("obs.npy", np.array(obs_list))
                print(velocity_list)
                print("done")
        if not get_data:
            if i == 1000:
                np.save("fix_position.npy", np.array(position_list))
                np.save("fix_velocity.npy", np.array(velocity_list))
                np.save("fix_obs.npy", np.array(obs_list))
                print("done")
                
            

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
