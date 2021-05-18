import numpy as np

from hrl_pybullet_envs.envs.gather.gather_base import GatherBulletEnv
from hrl_pybullet_envs.envs.gather.point_bot import PointBot


class PointGatherBulletEnv(GatherBulletEnv):
    def __init__(self,
                 n_food=8,
                 n_poison=8,
                 world_size=(15, 15),
                 n_bins=5,
                 sensor_range=20.,
                 sensor_span=np.pi,
                 robot_coll_dist=1,
                 robot_object_spacing=2.,
                 dying_cost=-10,
                 render=False,
                 use_sensor=True,
                 respawn=True,
                 debug=False):
        self.robot = PointBot()
        super().__init__(self.robot, n_food, n_poison, world_size, n_bins, sensor_range, sensor_span, robot_coll_dist,
                         robot_object_spacing, dying_cost, render, use_sensor, respawn, debug)
