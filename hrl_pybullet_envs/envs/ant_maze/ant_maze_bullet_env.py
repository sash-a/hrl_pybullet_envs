from typing import List

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.envs.ant_maze.maze_scene import MazeScene
from hrl_pybullet_envs.envs.ant_maze.maze_utils import pol2cart, Point, intersection, quadrant
from hrl_pybullet_envs.utils import debug_draw_point, PositionEncoding
from gym.spaces import Box
import gym


class AntMazeBulletEnv(AntBulletEnv):
    """
    At evaluation time, we evaluate the agent only on its ability to reach (0,16). We define a “success” as being within
    an L2 distance of 5 from the target on the ultimate step of the episode. - Data efficient HRL
    """
    eval_target = [6, -4]
    targets = [[6, 4], [0, 4], [-6, 4], eval_target]

    def __init__(self, n_bins: int = 8, sensor_range: float = 4, sensor_span: float = np.pi,
                 target_encoding: PositionEncoding = 0, tol=1, seed=None, debug=False):
        super().__init__()
        self.robot.start_pos_x, self.robot.start_pos_y, self.robot.start_pos_z = -6, -4, 0.25

        self.n_bins = n_bins
        self.sensor_range = float(sensor_range)
        self.sensor_span = sensor_span

        self.tol = tol
        if isinstance(target_encoding, int):
            target_encoding = PositionEncoding(target_encoding)
        self.target_encoding = target_encoding
        self.target: np.ndarray = np.array(AntMazeBulletEnv.eval_target)

        self.mpi_common_rand, _ = gym.utils.seeding.np_random(seed)

        self.debug = debug

        self.action_space = self.robot.action_space

        self.observation_space = Box(-np.inf, np.inf, shape=(self.robot.observation_space.shape[0] - 2 + n_bins + 2,))

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = MazeScene(bullet_client, 9.8, 0.0165 / 4, 4)
        return self.stadium_scene

    def _get_obs(self, ant_obs):
        wall_obs = self.scene.sense_walls(self.n_bins, self.sensor_span, self.sensor_range,
                                          self.robot.body_real_xyz[:2], self.robot_body.pose().rpy()[2], self.debug)

        target_obs = []
        vec_to_target = self.target - self.robot_body.pose().xyz()[:2]

        if self.target_encoding == PositionEncoding.normed_vec:
            target_obs = vec_to_target / np.linalg.norm(vec_to_target)
        elif self.target_encoding == PositionEncoding.angle:
            angle_to_target = np.arctan2(*vec_to_target[::-1]) - self.robot_body.pose().rpy()[2]
            target_obs = [np.sin(angle_to_target), np.cos(angle_to_target)]

        return np.concatenate((target_obs, [ant_obs[0]], ant_obs[3:], wall_obs))
        # return np.concatenate((ant_obs, wall_obs))

    def step(self, a):
        if self.debug:
            debug_draw_point(self.scene._p, *self.target, colour=[0.1, 0.5, 0.7])
        ant_obs, ant_rew, d, i = super().step(a)
        obs = self._get_obs(ant_obs)

        dist = np.linalg.norm(self.target - self.robot_body.pose().xyz()[:2])
        rew = -dist / self.scene.dt

        # TODO possibly cache this to make it faster
        if dist < self.tol:
            rew += 10000
            d = True

        return obs, rew, d, i

    def reset(self):
        if not hasattr(self, '_p'):
            super().reset()

        start_xyz = [self.robot.start_pos_x, self.robot.start_pos_y, self.robot.start_pos_z]
        self.target = AntMazeBulletEnv.targets[self.mpi_common_rand.randint(0, len(AntMazeBulletEnv.targets))]
        # self._p.resetBasePositionAndOrientation(self.robot.objects[0], start_xyz, [0, 0, 0, 1])
        super().reset()
        self._p.resetBasePositionAndOrientation(self.robot.objects[0], start_xyz, [0, 0, 0, 1])
        self.robot.robot_specific_reset(self.scene._p)
        ant_obs = self.robot.calc_state()

        return self._get_obs(ant_obs)
