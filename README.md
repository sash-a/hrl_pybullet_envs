## Hierarchical Reinforcement envs in pybullet

This package was created because all of the HRL locomotion envs are only available in mujoco. This is an implementation of as many as possible in pybullet. 

### Install
`pip install pybullet hrl_pybullet_envs`  
This project requires [pybullet-gym](https://github.com/benelot/pybullet-gym/) which must be installed along side this package.


### Envs:
* AntGatherBulletEnv-v0
* AntMazeBulletEnv-v0
* AntMjBulletEnv-0
* AntFlagrunBulletEnv-v0
* PointGatherBulletEnv-v0

### Example
Also see [this notebook](https://colab.research.google.com/drive/17FX7UM1-DDb3oxg1ei64dw9Xa6JFE_zF?usp=sharing)
```
import hrl_pybullet_envs
import gym
import numpy as np

env = gym.make('AntGatherBulletEnv-v0')
env.render()
ob = env.reset()
tot_rew = 0

for i in range(1000):
  # Take random actions
  ob, rew, done, _ = env.step(np.random.uniform(-1, 1, env.action_space.shape))
  tot_rew += rew

  if done: break

print(f'Achieved total reward of: {tot_rew}')
```
