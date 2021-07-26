from os.path import join
from typing import List

import numpy as np
import pybullet
from pybullet_envs.scene_abstract import Scene

from hrl_pybullet_envs.assets import assets_dir
from hrl_pybullet_envs.envs.intersection_utils import Point, pol2cart, quadrant, inf_intersection
from hrl_pybullet_envs.utils import debug_draw_point


class SizeableEnclosedScene(Scene):
    def __init__(self, bullet_client, gravity, timestep, frame_skip, world_size: tuple, world_center: tuple = (0, 0)):
        """
        Flat scene enclosed with walls on all sides

        :param world_size: tuple of the x and y dimensions of the world. world_size+world_center must be less than 50
        :param world_center: tuple of the center position of the world. world_size+world_center must be less than 50
        """
        super().__init__(bullet_client, gravity, timestep, frame_skip)
        self.loaded = False
        self.multiplayer = False
        self.size = world_size
        self.center = world_center
        self.ground_plane_mjcf = []

        world_p1 = Point(world_size[0] / 2 + world_center[0], world_size[1] / 2 + world_center[1])
        world_p2 = Point(-world_size[0] / 2 + world_center[0], -world_size[1] / 2 + world_center[1])

        self.world_bounds = [(world_p1, Point(world_p2.x, world_p1.y)),
                             (world_p1, Point(world_p1.x, world_p2.y)),
                             (world_p2, Point(world_p2.x, world_p1.y)),
                             (world_p2, Point(world_p1.x, world_p2.y))]
        self.box_bounds = []  # bounds of a single box shaped obstacle in the world

    bounds = property(lambda self: self.world_bounds + self.box_bounds)

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if not self.loaded:
            self.loaded = True
            self.ground_plane_mjcf += [
                self._p.loadURDF(join(assets_dir, 'plane.xml'), (0 + self.center[0], 0 + self.center[1], 0))]
            self.ground_plane_mjcf += [
                self._p.loadURDF(join(assets_dir, 'wall.xml'),
                                 (self.center[0], self.size[1] / 2 + self.center[1], 2.5))]
            self.ground_plane_mjcf += [
                self._p.loadURDF(join(assets_dir, 'wall.xml'),
                                 (self.center[0], -self.size[1] / 2 + self.center[1], 2.5))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'),
                                                        (self.size[0] / 2 + self.center[0], 0 + self.center[1], 2.5),
                                                        pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'),
                                                        (-self.size[0] / 2 + self.center[0], 0 + self.center[1], 2.5),
                                                        pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)))]

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

    def sense_walls(self, s_bins, s_span, s_range, robot_pos, rob_yaw, debug=False) -> List[float]:
        sensor = [0 for _ in range(s_bins)]
        # robot_pos = self.robot_body.pose().xyz()[:2]
        for i in range(s_bins):
            # each loop sensor is 1 more 'nth' of the sensor span - special case for 2pi because 0 == 2pi
            if s_span == 2 * np.pi:
                polar = (s_range, np.pi / 2 + rob_yaw + ((i + 1) / s_bins) * s_span)
            else:
                polar = (s_range, np.pi / 2 + rob_yaw + (i / (s_bins - 1)) * s_span)
            sensor_vec = np.array(robot_pos) + pol2cart(*polar)  # line coming from robot in dir
            # What quadrant the sensor would be in if it started from origin, used to avoid sensing behind robot
            sensor_quadrant = quadrant(Point(*(np.array(sensor_vec) - robot_pos)))

            if debug:
                self._p.addUserDebugLine([*robot_pos, 0], [*sensor_vec, 0], lifeTime=0.1)

            for line in self.bounds:  # Start and end points of bounding boxes around scene obstacles
                # find intersection of sensor line and current bounding line
                inter = inf_intersection(Point(*robot_pos), Point(*sensor_vec), *line)
                if inter is None:  # no intersection
                    continue

                dist = np.linalg.norm(robot_pos - np.array([*inter]))
                if dist > s_range:  # too far
                    continue

                if sensor_quadrant != quadrant(Point(*(np.array([*inter]) - robot_pos))):
                    continue  # intersection is directly behind the sensor

                if debug:
                    debug_draw_point(self._p, *inter)

                sensor[i] = max(sensor[i], 1. - dist / s_range)  # closest sensor value is added to list

        return sensor
