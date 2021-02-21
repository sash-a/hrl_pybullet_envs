import gym

from hrl_pybullet_envs.ant_maze.ant_maze_env import AntMazeBulletEnv, AntMazeBulletEnv

gym.envs.register(
    id='AntMazeBulletEnv-v0',
    entry_point=f'{AntMazeBulletEnv.__module__}:{AntMazeBulletEnv.__name__}',
    max_episode_steps=2000,
)
