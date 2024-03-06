# Copyright (c) 2022-2024, The ORBIT Project Developers.
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
from omni.isaac.orbit_assets.unitree import ALIENGO_Z1_CFG  # isort: skip
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm

from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.sensors import RayCaster
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp

# define specific functions here
# rewards
def base_height_l1(
        env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    target_height = 0.4
    asset: RigidObject = env.scene[asset_cfg.name]
    height_scanner: RayCaster = env.scene["height_scanner"]
    ground_height = height_scanner.data.ray_hits_w[:, :, 2].mean(dim=1)
    relative_height = asset.data.root_pos_w[:, 2] - ground_height
    return torch.abs(relative_height - target_height)

# terminations
def base_height_terminate(
    env: RLTaskEnv,
) -> torch.Tensor:
    asset = env.scene["robot"]
    height_scanner: RayCaster = env.scene["height_scanner"]
    ground_height = height_scanner.data.ray_hits_w[:, :, 2].mean(dim=1)
    relative_height = asset.data.root_pos_w[:, 2] - ground_height
    return relative_height < 0.3
    

@configclass
class AliengoZ1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = ALIENGO_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # randomization
        self.randomization.push_robot = None
        self.randomization.add_base_mass.params["mass_range"] = (-1.0, 3.0)
        self.randomization.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.randomization.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
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

        # self-defined reward function
        # self.rewards.base_height = RewTerm(func=base_height_l1, weight=-0.2)
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )
        # self.rewards.undesired_contacts = RewTerm(
        #     func=mdp.undesired_contacts,
        #     weight=-1.0,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
        # )
        self.rewards.undesired_contacts = None
        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts.weight = -0.01
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.feet_air_time=None
        # self.rewards.undesired_contacts.weight = -0.01
        # self.rewards.dof_torques_l2=None
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.75
        # self.rewards.dof_acc_l2=None
        # self.rewards.ang_vel_xy_l2=None
        # self.rewards.action_rate_l2=None
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*trunk"
        # self.terminations.base_height = DoneTerm(
        #     func=base_height_terminate, 
        # )
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None

@configclass
class AliengoZ1RoughEnvCfg_PLAY(AliengoZ1RoughEnvCfg):
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
