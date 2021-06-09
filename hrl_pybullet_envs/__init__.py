import gym

from hrl_pybullet_envs.envs.ant_flagrun.ant_flagrun_env import AntFlagrunBulletEnv
from hrl_pybullet_envs.envs.ant_maze.ant_maze_env import AntMazeMjEnv
from hrl_pybullet_envs.envs.gather.ant_gather_env import AntGatherBulletEnv
from hrl_pybullet_envs.envs.gather.point_gather_env import PointGatherBulletEnv

__all__ = [AntGatherBulletEnv, AntMazeMjEnv, AntFlagrunBulletEnv, PointGatherBulletEnv]

for env in __all__:
    gym.envs.register(
        id=f'{env.__name__}-v0',
        entry_point=f'{env.__module__}:{env.__name__}',
        max_episode_steps=2000,
    )
