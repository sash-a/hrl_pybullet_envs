import inspect
from os import path

import gym

from hrl_pybullet_envs.ant_maze.ant_maze_env import AntMazeBulletEnv


def classpath(cls):
    return path.relpath(inspect.getfile(cls)).replace('/', '.')[:-3]


gym.envs.register(
    id='AntMazeBulletEnv-v0',
    entry_point=f"{classpath(AntMazeBulletEnv)}:AntMazeBulletEnv",
    max_episode_steps=2000,
)
