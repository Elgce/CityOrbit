# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Widow-Go1-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.WidowGo1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.WidowGo1FlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Widow-Go1-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.WidowGo1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.WidowGo1FlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Widow-Go1-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.WidowGo1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.WidowGo1RoughPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Widow-Go1-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.WidowGo1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.WidowGo1RoughPPORunnerCfg,
    },
)
