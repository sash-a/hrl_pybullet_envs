import math
import warnings

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.envs.ant_maze.maze_scene import MazeScene
from hrl_pybullet_envs.envs.intersection_utils import Point, segment_intersection
from hrl_pybullet_envs.utils import debug_draw_point, PositionEncoding
from gym.spaces import Box
import gym

_eval_target = [-2, 4]
_targets = ([2, -3], [2, 0], [2, 3], _eval_target)


class AntMazeBulletEnv(AntBulletEnv):
    """
    At evaluation time, we evaluate the agent only on its ability to reach (0,16). We define a “success” as being within
    an L2 distance of 5 from the target on the ultimate step of the episode. - Data efficient HRL
    """

    def __init__(self, n_bins: int = 10, sensor_range: float = 5, sensor_span: float = 2 * np.pi, targets=_targets,
                 target_encoding: PositionEncoding = 0, sense_target=False, sense_walls=True, done_at_target=True,
                 max_steps=-1, tol=1.5, inner_rew_weight=0, targ_dist_rew=False, seed=None, debug=0):
        super().__init__()
        self.robot.start_pos_x, self.robot.start_pos_y, self.robot.start_pos_z = -2, -5, 0.25

        self.n_bins = n_bins
        self.sensor_range = float(sensor_range)
        self.sensor_span = sensor_span

        self.targets = targets
        self.sense_walls = sense_walls
        self.sense_target = sense_target
        self.done_at_target = done_at_target
        self.max_steps = max_steps
        self.t = 0
        self.tol = tol
        self.inner_rew_weight = inner_rew_weight
        self.targ_dist_rew = targ_dist_rew

        if isinstance(target_encoding, int):
            target_encoding = PositionEncoding(target_encoding)
        self.target_encoding = target_encoding
        self.target: np.ndarray = np.array(_eval_target)

        self.rs, _ = gym.utils.seeding.np_random(seed)

        self.debug = debug

        self.action_space = self.robot.action_space

        target_obs = n_bins if sense_target else 2
        walls_obs = n_bins if sense_walls else 0
        shape = (self.robot.observation_space.shape[0] - 2 + walls_obs + target_obs,)
        self.observation_space = Box(-np.inf, np.inf, shape=shape)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = MazeScene(bullet_client, 9.8, 0.0165 / 4, 4)
        return self.stadium_scene

    def _get_obs(self, ant_obs):
        wall_obs = []
        if self.sense_walls:
            wall_obs = self.scene.sense_walls(self.n_bins, self.sensor_span, self.sensor_range,
                                              self.robot.body_real_xyz[:2], self.robot_body.pose().rpy()[2],
                                              self.debug > 1)

        if self.sense_target:
            target_obs = self.get_target_sensor_obs()
        else:
            target_obs = self.get_target_vec_obs()

        return np.concatenate(([ant_obs[0]], ant_obs[3:], target_obs, wall_obs))

    def step(self, a):
        self.t += 1
        if self.debug > 0:
            debug_draw_point(self.scene._p, *self.target, colour=[0.1, 0.5, 0.7])
        ant_obs, inner_rew, d, i = super().step(a)
        obs = self._get_obs(ant_obs)

        rew = inner_rew * self.inner_rew_weight

        if self.robot.walk_target_dist < self.tol:
            if self.done_at_target or (not self.done_at_target and self.t == self.max_steps - 1):
                rew += 1
                d = True

        if self.t == self.max_steps - 1:
            d = True

        if self.targ_dist_rew and d:  # rewarding based on distance to target on final step
            rew -= self.robot.walk_target_dist

        return obs, rew, d, i

    def seed(self, seed=None):
        super().seed(seed)
        self.rs, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        if not hasattr(self, '_p'):
            super().reset()

        self.t = 0
        start_xyz = [self.robot.start_pos_x, self.robot.start_pos_y, self.robot.start_pos_z]
        self.target = self.targets[self.rs.randint(0, len(self.targets))]
        super().reset()

        # Allows for correct inner reward
        self.walk_target_x, self.walk_target_y = self.target
        self.robot.walk_target_x, self.robot.walk_target_y = self.target

        self._p.resetBasePositionAndOrientation(self.robot.objects[0], start_xyz, [0, 0, 0, 1])
        self.robot.robot_specific_reset(self.scene._p)
        ant_obs = self.robot.calc_state()

        return self._get_obs(ant_obs)

    def get_target_vec_obs(self):
        target_obs = []
        vec_to_target = self.target - self.robot_body.pose().xyz()[:2]

        if self.target_encoding == PositionEncoding.normed_vec:
            target_obs = vec_to_target / np.linalg.norm(vec_to_target)
        elif self.target_encoding == PositionEncoding.angle:
            angle_to_target = np.arctan2(*vec_to_target[::-1]) - self.robot_body.pose().rpy()[2]
            target_obs = [np.sin(angle_to_target), np.cos(angle_to_target)]

        return target_obs

    def get_target_sensor_obs(self) -> np.ndarray:
        """compute sensor readings for target"""
        if self.n_bins <= 0: return np.array([])
        # first, obtain current orientation
        readings = np.zeros(self.n_bins)
        rob_pos = Point(*self.robot_body.pose().xyz()[:2])
        target_pos = Point(*self.target)
        ori = self.robot_body.pose().rpy()[2]  # main change from rllab, this is the yaw of the robot
        bin_res = self.sensor_span / self.n_bins

        if self.robot.walk_target_dist > self.sensor_range:  # only include readings for objects within range
            return readings

        for p1, p2 in self.scene.box_bounds:  # check if the box occludes the goal
            if segment_intersection(rob_pos, target_pos, p1, p2):
                return readings

        # it is within distance and not occluded
        angle = math.atan2(target_pos.y - rob_pos.y, target_pos.x - rob_pos.x) - ori
        if math.isnan(angle):
            warnings.warn('angle is nan')
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle = angle - 2 * math.pi
        if angle < -math.pi:
            angle = angle + 2 * math.pi

        # outside of sensor span - skip this
        half_span = self.sensor_span * 0.5
        if abs(angle) > half_span:
            return readings
        bin_number = int((angle + half_span) / bin_res)
        intensity = 1.0 - self.robot.walk_target_dist / self.sensor_range
        readings[bin_number] = intensity

        # useful debug
        if self.debug > 1 and intensity > 0:
            colour = [0, 0, 1]
            self._p.addUserDebugLine(self.robot_body.pose().xyz(), [target_pos.x, target_pos.y, 0.25], lifeTime=0.5,
                                     lineColorRGB=colour)
            self._p.addUserDebugText(f'{intensity:0.2f}', [target_pos.x, target_pos.y, 0.25], lifeTime=0.5,
                                     textColorRGB=colour)

        return readings
