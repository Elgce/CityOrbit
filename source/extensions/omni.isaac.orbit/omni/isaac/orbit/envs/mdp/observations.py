# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import RayCaster
from omni.isaac.orbit.envs.mdp.traj_generator import TrajGenerator
from omni.isaac.orbit.utils import torch_utils
if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv, RLTaskEnv

"""
Root state.
"""


def base_pos_z(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


# define observations for h1
def h1_root_pos(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :]

def h1_root_rot(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_quat_w[:, :]

def h1_root_lin_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, :]

def h1_root_ang_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w[:, :]

def h1_dof_pos(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos

def h1_dof_vel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel

def h1_default_dof_pos(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_joint_pos

def process_h1_obs(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    root_pos = h1_root_pos(env, asset_cfg)
    root_rot = h1_root_rot(env, asset_cfg)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    root_lin_vel = h1_root_lin_vel(env, asset_cfg)
    root_ang_vel = h1_root_ang_vel(env, asset_cfg)

    local_root_vel = torch_utils.my_quat_rotate(heading_rot, root_lin_vel)
    local_root_ang_vel = torch_utils.my_quat_rotate(heading_rot, root_ang_vel)
    dof_pos = h1_dof_pos(env, asset_cfg)
    dof_vel = h1_dof_vel(env, asset_cfg)
    default_dof_pos = h1_default_dof_pos(env, asset_cfg)
    local_dof_pos = dof_pos - default_dof_pos
    gravity = projected_gravity(env, asset_cfg)
    gravity = torch_utils.my_quat_rotate(heading_rot, gravity)
    
    # dof_pos_limit = asset.data.soft_joint_pos_limits
    obs = torch.cat([local_root_vel, local_root_ang_vel, local_dof_pos, dof_vel, gravity], dim=-1)
    return obs
    
    

def h1_traj_obs(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        process_time = torch.zeros(env.num_envs, device=env.device)
    else:
        process_time = env.episode_length_buf
    traj_samples = fetch_traj_samples(env, process_time, asset_cfg)
    return traj_samples

def fetch_traj_samples(env: BaseEnv, progress_buf, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ):
    asset: Articulation = env.scene[asset_cfg.name]
    numTrajSamples = 10
    trajSampleTimestep = 0.5
    timestep_beg = progress_buf * 0.02
    timesteps = torch.arange(numTrajSamples, dtype=torch.float, device=env.device)
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    timesteps = timesteps * trajSampleTimestep
    traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps
    env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)
    traj_gen = TrajGenerator(
        env.num_envs, 300, 101,
        "cuda:0", 2.0,
        0, 2, 1, 0.02
    )
    root_pos = asset.data.root_pos_w
    traj_gen.reset(torch.arange(env.num_envs), root_pos)
    traj_samples_flat = traj_gen.calc_pos(env_ids_tiled.flatten(), traj_timesteps.flatten())
    traj_samples = torch.reshape(traj_samples_flat, shape=(env_ids.shape[0], numTrajSamples, traj_samples_flat.shape[-1]))
    
    root_rot = asset.data.root_quat_w
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(heading_rot.unsqueeze(-2), (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]))
    heading_rot_exp = torch.reshape(heading_rot_exp, (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]))
    traj_samples_delta = traj_samples - root_pos.unsqueeze(-2)
    traj_samples_delta_flat = torch.reshape(traj_samples_delta, (traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
                                                                 traj_samples_delta.shape[2]))

    local_traj_pos = torch_utils.my_quat_rotate(heading_rot_exp, traj_samples_delta_flat)
    local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(local_traj_pos, (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]))
    return obs
    # return traj_samples
    
"""
Joint state.
"""


def joint_pos_rel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.joint_pos - asset.data.default_joint_pos)[:, :19]


def joint_pos_norm(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
    )


def joint_vel_rel(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.joint_vel - asset.data.default_joint_vel)[:, :19]


"""
Sensors.
"""


def height_scan(env: BaseEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.
    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


def body_incoming_wrench(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.body_ids]
    return link_incoming_forces.view(env.num_envs, -1)


"""
Actions.
"""


def last_action(env: BaseEnv) -> torch.Tensor:
    """The last input action to the environment."""
    return env.action_manager.action


"""
Commands.
"""


def generated_commands(env: RLTaskEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)
