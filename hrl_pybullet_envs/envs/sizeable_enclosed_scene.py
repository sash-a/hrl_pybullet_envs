from os.path import join

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
