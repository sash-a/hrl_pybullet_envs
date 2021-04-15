import gym

from hrl_pybullet_envs.envs.ant_flagrun.env import AntFlagrunBulletEnv
from hrl_pybullet_envs.envs.ant_gather.env import AntGatherBulletEnv
from hrl_pybullet_envs.envs.ant_maze.env import AntMazeBulletEnv

__all__ = [AntGatherBulletEnv, AntMazeBulletEnv, AntFlagrunBulletEnv]

for env in __all__:
    gym.envs.register(
        id=f'{env.__name__}-v0',
        entry_point=f'{env.__module__}:{env.__name__}',
        max_episode_steps=2000,
    )
