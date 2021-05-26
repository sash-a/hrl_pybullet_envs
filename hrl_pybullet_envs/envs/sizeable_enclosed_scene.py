import math
from os.path import join
from typing import List, Tuple

import numpy as np
import pybullet
from pybullet_envs.scene_abstract import Scene

from hrl_pybullet_envs.assets import assets_dir


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

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if not self.loaded:
            self.load_assets()

    def load_assets(self):
        self.loaded = True
        self.ground_plane_mjcf += [
            self._p.loadURDF(join(assets_dir, 'plane.xml'), (0 + self.center[0], 0 + self.center[1], 0))]
        self.ground_plane_mjcf += [
            self._p.loadURDF(join(assets_dir, 'wall.xml'), (self.center[0], self.size[1] / 2 + self.center[1], 2.5))]
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


class VecSquareEnclosedScene(Scene):
    multiplayer = False

    def __init__(self, bullet_client, gravity, timestep, frame_skip, area_size: float, n: int):
        super().__init__(bullet_client, gravity, timestep, frame_skip)

        self.loaded = False

        self.area_size = area_size
        self.world_centers: List[Tuple[float, float]] = self.calc_centers(n, area_size)
        self.n = n

        self.ground_plane_mjcf = []

    def actor_introduce(self, robot):
        pass
        # Scene.actor_introduce(self, robot)
        # i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        # robot.robot_body.pose().move_xyz(self.world_centers[robot.player_n])

    @staticmethod
    def calc_centers(n, area_size):
        world_centers = []
        spawned_count = 0
        n_rows = math.ceil(np.sqrt(n))
        for x in range(n_rows):
            for y in range(n_rows):
                world_centers += [((x - n_rows / 2) * area_size, (y - n_rows / 2) * area_size)]
                spawned_count += 1

                if spawned_count >= n:
                    break
            if spawned_count >= n:
                break

        return world_centers

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if not self.loaded:
            self.load_assets()

    def load_assets(self):
        self.loaded = True
        floor = self._p.loadURDF(join(assets_dir, 'plane.xml'), (0, 0, 0))
        self.ground_plane_mjcf += [floor]

        spawned_count = 0
        n_rows = math.ceil(np.sqrt(self.n))
        # for x in range(n_rows):
        #     for y in range(n_rows):
        #         self.world_centers += [((x - n_rows / 2) * self.area_size, (y - n_rows / 2) * self.area_size)]
        #         self.load_enclosure(self.world_centers[-1])
        #         spawned_count += 1
        #
        #         if spawned_count >= self.n:
        #             break
        #     if spawned_count >= self.n:
        #         break
        for center in self.world_centers:
            self.load_enclosure(center)

    def load_enclosure(self, center):
        wall_height = 2

        half_extents = [self.area_size / 2, 0.1, wall_height]

        up = (center[0], self.area_size / 2. + center[1], wall_height / 2.)
        down = (center[0], -self.area_size / 2. + center[1], wall_height / 2.)
        right = (self.area_size / 2. + center[0], center[1], wall_height / 2.)
        left = (-self.area_size / 2. + center[0], center[1], wall_height / 2.)

        top_rot = pybullet.getQuaternionFromEuler((0, 0, np.pi))
        side_rot = pybullet.getQuaternionFromEuler((0, 0, np.pi / 2))

        for pos, rot in zip([up, down, right, left], [top_rot, top_rot, side_rot, side_rot]):
            col = self._p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half_extents)
            viz = self._p.createVisualShape(pybullet.GEOM_BOX, halfExtents=half_extents)
            self.ground_plane_mjcf += [self._p.createMultiBody(0, col, viz, pos, rot)]
