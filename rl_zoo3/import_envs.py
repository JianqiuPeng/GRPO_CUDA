from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register, register_envs

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs_gymnasium
except ImportError:
    pass

try:
    import ale_py

    # no-op
    gym.register_envs(ale_py)
except ImportError:
    pass

try:
    import highway_env
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import custom_envs
except ImportError:
    pass

try:
    import gym_donkeycar
except ImportError:
    pass

try:
    import panda_gym
except ImportError:
    pass

try:
    import rocket_lander_gym
except ImportError:
    pass

try:
    import minigrid
except ImportError:
    pass


# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )


import os
config_folders = os.listdir("custom_envs/configs/exp-change_antenna_num")
for folder in config_folders:
    config_folder_path = os.path.join("custom_envs/configs/exp-change_antenna_num", folder)
    register(
        id="MISOEnv-" + folder,
        entry_point="custom_envs.MISOenv:MISOEnvWrapper",  # Replace with the actual module path
        kwargs={"config_folder_path": config_folder_path},
    )
