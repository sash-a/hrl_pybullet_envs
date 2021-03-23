from os.path import join
from typing import Tuple, List

import gym
import numpy as np
import pybullet
from numpy.random import RandomState

from hrl_pybullet_envs.assets import assets_dir
from hrl_pybullet_envs.envs.sizeable_enclosed_scene import SizeableEnclosedScene


class GatherScene(SizeableEnclosedScene):
    no_rot = pybullet.getQuaternionFromEuler((0, 0, 0))
    fake_kill_pos = [100, 0, -10]

    def __init__(self, bullet_client, gravity, timestep, frame_skip, world_size: tuple, n_food: int, n_poison: int,
                 robot_object_spacing: float = 2, respawn: bool = True):
        """
        Creates an gather scene with food and poison spawned in on episode restart

        :param world_size: tuple of the x and y dimensions of the world must be less than 50
        """
        super().__init__(bullet_client, gravity, timestep, frame_skip, world_size)
        self.food = {}
        self.poison = {}
        self.n_food = n_food
        self.n_poison = n_poison

        self.spacing = robot_object_spacing
        self.respawn = respawn

        self.rs, _ = gym.utils.seeding.np_random(None)

    all_items = property(lambda self: {**self.food, **self.poison})

    def seed(self, seed):
        self.rs, _ = gym.utils.seeding.np_random(seed)

    def episode_restart(self, bullet_client):
        super().episode_restart(bullet_client)

        for i in range(self.n_food - len(self.food)):
            self.spawn_random_food([0, 0])
        for i in range(self.n_poison - len(self.poison)):
            self.spawn_random_poison([0, 0])

        for obj_id in list(self.food.keys()):
            self.move_food_random(obj_id, [0, 0])

        for obj_id in list(self.poison.keys()):
            self.move_poison_random(obj_id, [0, 0])

    def _random_on_plane(self, avoid_xy: List[float] = None) -> list:
        if avoid_xy is None:  # avoid a point too far away to matter
            avoid_xy = [-10000, -10000]
        avoid_xy = np.array(avoid_xy)
        size = np.array(self.size) - 1

        pos = self.rs.rand(2) * size - size / 2
        while np.linalg.norm(avoid_xy - pos) < self.spacing:
            pos = self.rs.rand(2) * size - size / 2

        return pos.tolist() + [0.1]

    def _spawn_random_on_plane(self, urdf_path: str, avoid_xy: List[float]) -> Tuple[int, List[float]]:
        pos = self._random_on_plane(avoid_xy)
        obj_id = self._p.loadURDF(urdf_path, basePosition=pos)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, obj_id)

        return obj_id, pos

    def _move_random_on_plane(self, obj_id: int, avoid_xy: List[float]) -> List[float]:
        pos = self._random_on_plane(avoid_xy)
        self._p.resetBasePositionAndOrientation(obj_id, pos, self.no_rot)

        return pos

    def spawn_random_food(self, avoid_xy: List[float]):
        food_id, pos = self._spawn_random_on_plane(join(assets_dir, 'food.xml'), avoid_xy)
        self.food[food_id] = pos
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, food_id)

    def spawn_random_poison(self, avoid_xy: List[float]):
        poison_id, pos = self._spawn_random_on_plane(join(assets_dir, 'poison.xml'), avoid_xy)
        self.poison[poison_id] = pos
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, poison_id)

    def move_food_random(self, obj_id: int, avoid_xy: List[float]):
        pos = self._move_random_on_plane(obj_id, avoid_xy)
        self.food[obj_id] = pos

    def move_poison_random(self, obj_id: int, avoid_xy: List[float]):
        pos = self._move_random_on_plane(obj_id, avoid_xy)
        self.poison[obj_id] = pos

    def reward_collision(self, obj_id: int, agent_xyz: List[float]):
        """returns -1 if id was poison and 1 if id was food, 0 otherwise"""
        if obj_id in self.food:
            if self.respawn:
                self.move_food_random(obj_id, agent_xyz[:2])
            else:  # move away instead of deleting
                self._p.resetBasePositionAndOrientation(obj_id, self.fake_kill_pos, self.no_rot)
                self.food[obj_id] = self.fake_kill_pos
            rew = 1
        elif obj_id in self.poison:
            if self.respawn:
                self.move_poison_random(obj_id, agent_xyz[:2])
            else:  # move away instead of deleting
                self._p.resetBasePositionAndOrientation(obj_id, self.fake_kill_pos, self.no_rot)
                self.poison[obj_id] = self.fake_kill_pos
            rew = -1
        else:
            rew = 0

        return rew

