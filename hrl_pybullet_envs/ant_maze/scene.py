from os.path import join

import pybullet
from pybullet_envs.scene_abstract import Scene

from hrl_pybullet_envs.assets import assets_dir


class MazeScene(Scene):
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        super().__init__(bullet_client, gravity, timestep, frame_skip)
        self.loaded = False
        self.ground_plane_mjcf = []
        self.multiplayer = False

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if not self.loaded:
            self.loaded = True
            # Agent spawns at (0, 0, 0). Objects centered at (5, -7.5, 0) so that agent spawns in corner
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'plane.xml'), basePosition=(5, -7.5, 0))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'box.xml'), basePosition=(2, -7.5, 1.25))]

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
                self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

    def _spawn_random(self, urdf_path: str, bounds: tuple):
        pass

    def spawn_random_food(self):
        pass

    def spawn_random_poison(self):
        pass
