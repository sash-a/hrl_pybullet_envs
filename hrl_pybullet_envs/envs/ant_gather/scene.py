from os.path import join

import numpy as np
import pybullet

from hrl_pybullet_envs.assets import assets_dir
from hrl_pybullet_envs.envs.sizeable_enclosed_scene import SizeableEnclosedScene


class GatherScene(SizeableEnclosedScene):
    def __init__(self, bullet_client, gravity, timestep, frame_skip, world_size: tuple, n_food: int, n_poison: int):
        """
        Creates an gather scene with food and poison spawned in on episode restart

        :param world_size: tuple of the x and y dimensions of the world must be less than 50
        """
        super().__init__(bullet_client, gravity, timestep, frame_skip, world_size)
        self.food = {}
        self.poison = {}
        self.n_food = n_food
        self.n_poison = n_poison

        self.rs: np.random.RandomState = np.random.RandomState()

    def seed(self, rs: np.random.RandomState):
        self.rs = rs

    def episode_restart(self, bullet_client):
        super().episode_restart(bullet_client)

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

    # TODO:
    #  optional respawning
    #  don't spawn food to close to agent
    def destroy_and_respawn(self, obj_id):
        """returns -1 if id was poison and 1 if id was food."""
        if obj_id in self.food:
            self.food.pop(obj_id)
            self.spawn_random_food()
            self._p.removeBody(obj_id)
            rew = 1
        elif obj_id in self.poison:
            self.poison.pop(obj_id)
            self.spawn_random_poison()
            self._p.removeBody(obj_id)
            rew = -1
        else:
            rew = 0

        return rew

