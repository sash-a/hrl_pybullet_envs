from typing import List

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.envs.MjAnt import AntMjEnv
from hrl_pybullet_envs.envs.ant_maze.maze_scene import MazeScene
from hrl_pybullet_envs.envs.ant_maze.maze_utils import pol2cart, Point, intersects, intersection, quadrant


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
            polar = (self.sensor_range, ((i + 1) / self.n_bins) * self.sensor_span)
            cart = pol2cart(*polar)  # cartesian end point of sensor starting from origin
            sensor_end_point = robot_pos + cart
            quadrant_normed_sensor_end = quadrant(Point(*(np.array(sensor_end_point) - robot_pos)))
            if self.debug:
                self._p.addUserDebugLine([*robot_pos, 0], [*sensor_end_point, 0], lifeTime=0.5)

            for line in self.scene.bounds:
                inter = intersection(Point(*robot_pos), Point(*sensor_end_point), *line)
                if inter is None:  # no intersection leave reading at current
                    continue

                dist = np.linalg.norm(robot_pos - np.array([*inter]))
                if dist > self.sensor_range:  # too far leave reading at current
                    continue

                if quadrant_normed_sensor_end != quadrant(Point(*(np.array([*inter]) - robot_pos))):
                    continue  # the intersection is directly behind the sensor

                if self.debug:
                    self.debug_draw_point(inter)

                sensor[i] = max(sensor[i], 1. - dist / self.sensor_range)

        return sensor

    def debug_draw_point(self, p: Point):
        self.scene._p.addUserDebugLine([p.x + 0.5, p.y + 0.5, 0],
                                       [p.x - 0.5, p.y - 0.5, 0],
                                       lifeTime=0.5,
                                       lineColorRGB=[0.1, 0.2, 0.8])
        self.scene._p.addUserDebugLine([p.x + 0.5, p.y - 0.5, 0],
                                       [p.x - 0.5, p.y + 0.5, 0],
                                       lifeTime=0.5,
                                       lineColorRGB=[0.1, 0.2, 0.8])
