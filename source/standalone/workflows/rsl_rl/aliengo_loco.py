from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni_plat.runner.runner import SimulatorRunner
from omni_plat.runner.runner_rl import RLSimulatorRunner
from omni_plat.config import SimulatorConfig
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.prims import RigidPrim
from omni_plat.utils import get_pick_position
from omni_plat.robots.articulations.views.cabinet_view import CabinetView
from omni_plat.robots.articulations.views.door_view import DoorView
from omni_plat.robots.articulations.views.kujiale_view import KujialeView
from omni.isaac.core.utils.stage import add_reference_to_stage

from omni.isaac.core.articulations import Articulation

import matplotlib.pyplot as plt
import torch
import numpy as np
config = SimulatorConfig("./examples/aliengo_loco.yaml").config

runner = RLSimulatorRunner(config)

robot = runner._robots['aliengo']

robot_av = robot._robot._articulation_view
dof_names = robot_av._dof_names
for name in dof_names:
    idx = robot_av.get_dof_index(name)
    print(name, idx)


stage = runner._world.stage
runner._world._physics_context.set_gravity(0)

saved_action = np.load("./examples/action.npy")


# fig = plt.figure()

# x = np.linspace(0,1,200)
# for i, name in enumerate(dof_names[:4]):
#     y = saved_action[:,i]
#     plt.plot(x,y,label=name)

# plt.legend(loc="upper right")
# plt.show()

print(saved_action.shape)

start = 300
position_list = []
velocity_list = []
i = 0
try:
    while simulation_app.is_running():
        obs = runner._robots["aliengo"].get_obs()["aliengo_obs"]
        actions = {
            "aliengo": {
                    "quadruped_controller": (
                        np.zeros((1, 12))
                    )
                }
            }
        
        # with torch.inference_mode():
        #     actions = {
        #         "aliengo": {
        #             "quadruped_controller": (
        #                 runner._robots['aliengo']._controllers['quadruped_controller']._policy(torch.tensor(obs, dtype=torch.float32).to('cpu')).detach().numpy() * 0.25,
        #             )
        #         }
        #     }
        # p, o = robot_av.get_local_poses()
        # print(p)

        states = robot_av.get_joints_state()
        # print(states.positions, states.velocities)
        
        if i >= start:
            # print(i)
            actions['aliengo']['quadruped_controller'][0][:4] = saved_action[(i-start)%200][:4] * 0.25

            position_list.append(states.positions)
            velocity_list.append(states.velocities)
            
        else:
            actions['aliengo']['quadruped_controller'][0][:] = 0.
        obs = runner.step(actions=actions)
        
        # set base states
        robot_av.set_local_poses(np.array([[0,0,1]]),np.array([[1,0,0,0]]))
        robot_av.set_velocities(np.array([[0,0,0,0,0,0]]))

        
        if i == 900:
            break
        i += 1

finally:
    with open('positions.npy','wb') as fp:
        np.save(fp, np.array(position_list))
    
    with open('velocities.npy','wb') as fp:
        np.save(fp, np.array(velocity_list))
    
    pass

simulation_app.close()
