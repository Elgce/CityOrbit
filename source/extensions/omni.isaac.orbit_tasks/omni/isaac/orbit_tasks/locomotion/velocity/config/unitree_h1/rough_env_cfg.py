"""  """# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.orbit.envs import RLTaskEnv
##
# Pre-defined configs
##
from omni.isaac.orbit_assets.unitree import UNITREE_H1_CFG # isort: skip
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.sensors import RayCaster
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp

    

@configclass
class UnitreeH1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.robot = UNITREE_H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner = None
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        # reduce action scale
        self.actions.joint_pos.scale = 0.25
        self.randomization.push_robot = None
        self.randomization.add_base_mass.params["mass_range"] = (-1.0, 3.0)
        self.randomization.add_base_mass.params["asset_cfg"].body_names = "pelvis"
        self.randomization.base_external_force_torque.params["asset_cfg"].body_names = "pelvis"
        self.randomization.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.randomization.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"
        # self.terminations = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

@configclass
class UnitreeH1RoughEnvCfg_PLAY(UnitreeH1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.randomization.base_external_force_torque = None
        self.randomization.push_robot = None
