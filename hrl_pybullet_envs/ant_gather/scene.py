import warnings
from os.path import join

import numpy as np
import pybullet
from pybullet_envs.scene_abstract import Scene

from hrl_pybullet_envs.assets import assets_dir


class GatherScene(Scene):
    def __init__(self, bullet_client, gravity, timestep, frame_skip, n_food: int, n_poison: int, refresh: bool):
        super().__init__(bullet_client, gravity, timestep, frame_skip)
        self.loaded = False
        self.multiplayer = False
        self.size = (20, 20)

        self.ground_plane_mjcf = []
        self.food = {}
        self.poison = {}
        self.n_food = n_food
        self.n_poison = n_poison

        self.rs: np.random.RandomState = np.random.RandomState()

    def seed(self, seed):
        self.rs = np.random.RandomState(seed)

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if not self.loaded:
            self.loaded = True
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'plane.xml'), (0, 0, 0))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'), (0, 10, 2.5))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'), (0, -10, 2.5))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'), (10, 0, 2.5),
                                                        pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'), (-10, 0, 2.5),
                                                        pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)))]

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

        for i in range(self.n_food):
            self.spawn_random_food()
        for i in range(self.n_poison):
            self.spawn_random_poison()

    def _spawn_random_on_plane(self, urdf_path: str):
        size = np.array(self.size) - 1
        pos = (self.rs.rand(2) * size - size / 2).tolist() + [0.1]
        obj_id = self._p.loadURDF(urdf_path, basePosition=pos)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, obj_id)

        return obj_id, pos

    def spawn_random_food(self):
        food_id, pos = self._spawn_random_on_plane(join(assets_dir, 'food.xml'))
        self.food[food_id] = pos
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, food_id)

    def spawn_random_poison(self):
        poison_id, pos = self._spawn_random_on_plane(join(assets_dir, 'poison.xml'))
        self.poison[poison_id] = pos
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, poison_id)

    def destroy_and_respawn(self, obj_id):
        """returns -1 if id was poison and 1 if id was food."""
        if obj_id in self.food:
            self.food.pop(obj_id)
            self.spawn_random_food()
            rew = 1
        elif obj_id in self.poison:
            self.poison.pop(obj_id)
            self.spawn_random_poison()
            rew = -1
        else:
            warnings.warn(f'Tried to remove object that was neither a food nor a poison with id {obj_id}')
            rew = 0

        if rew != 0:
            self._p.removeBody(obj_id)
            print(f'destroyed and respawned! ({rew})')

        return rew

