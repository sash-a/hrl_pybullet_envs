from typing import List

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.envs.MjAnt import AntMjEnv
from hrl_pybullet_envs.envs.ant_maze.maze_scene import MazeScene
from hrl_pybullet_envs.envs.ant_maze.maze_utils import pol2cart, Point, intersection, quadrant
from hrl_pybullet_envs.utils import debug_draw_point


class AntMazeMjEnv(AntMjEnv):
    """
    At evaluation time, we evaluate the agent only on its ability to reach (0,16). We define a “success” as being within
    an L2 distance of 5 from the target on the ultimate step of the episode. - Data efficient HRL
    """

    def __init__(self, n_bins, sensor_range, sensor_span, debug=False):
        super().__init__(start_pos=(8, 8))
        self.walk_target_x = 0
        self.walk_target_y = -15

        self.n_bins = n_bins
        self.sensor_range = float(sensor_range)
        self.sensor_span = sensor_span

        self.debug = debug

        # TODO eval and random target positions

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = MazeScene(bullet_client, 9.8, 0.0165 / 4, 4, (20, 20))
        return self.stadium_scene

    def step(self, a):
        self.sense_walls()
        return super().step(a)

    def reset(self):
        return super().reset()

    def sense_walls(self) -> List[float]:
        sensor = [0 for _ in range(self.n_bins)]
        robot_pos = self.robot_body.pose().xyz()[:2]
        for i in range(self.n_bins):
            # each loop sensor is 1 more 'nth' of the sensor span - special case for 2pi because 0 == 2pi
            if self.sensor_span == 2 * np.pi:
                polar = (self.sensor_range, ((i + 1) / (self.n_bins)) * self.sensor_span)
            else:
                polar = (self.sensor_range, (i / (self.n_bins - 1)) * self.sensor_span)
            sensor_vec = robot_pos + pol2cart(*polar)  # line coming from robot in dir
            # What quadrant the sensor would be in if it started from origin, used to avoid sensing behind robot
            sensor_quadrant = quadrant(Point(*(np.array(sensor_vec) - robot_pos)))

            if self.debug:
                self._p.addUserDebugLine([*robot_pos, 0], [*sensor_vec, 0], lifeTime=0.5)

            for line in self.scene.bounds:  # Start and end points of bounding boxes around scene obstacles
                # find intersection of sensor line and current bounding line
                inter = intersection(Point(*robot_pos), Point(*sensor_vec), *line)
                if inter is None:  # no intersection
                    continue

                dist = np.linalg.norm(robot_pos - np.array([*inter]))
                if dist > self.sensor_range:  # too far
                    continue

                if sensor_quadrant != quadrant(Point(*(np.array([*inter]) - robot_pos))):
                    continue  # intersection is directly behind the sensor

                if self.debug:
                    debug_draw_point(self.scene._p, *inter)

                sensor[i] = max(sensor[i], 1. - dist / self.sensor_range)  # closest sensor value is added to list

        return sensor
