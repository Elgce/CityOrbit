# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.sensors import RayCaster
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
from .rough_env_cfg import UnitreeH1RoughEnvCfg
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm

def base_height_l1(
        env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    target_height = 0.4
    asset: RigidObject = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2]
    return torch.abs(relative_height - target_height)

# terminations
def base_height_terminate(
    env: RLTaskEnv,
) -> torch.Tensor:
    asset = env.scene["robot"]
    relative_height = asset.data.root_pos_w[:, 2]
    return relative_height < 0.3

@configclass
class UnitreeH1FlatEnvCfg(UnitreeH1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.rewards.base_height = RewTerm(func=base_height_l1, weight=-0.2)


        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 1.0

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.terminations.base_height = DoneTerm(
            func=base_height_terminate, 
        )


class UnitreeH1FlatEnvCfg_PLAY(UnitreeH1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
