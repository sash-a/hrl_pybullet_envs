from os.path import join

import numpy as np
import pybullet
from pybullet_envs.scene_abstract import Scene

from hrl_pybullet_envs.assets import assets_dir


class GatherScene(Scene):
    def __init__(self, bullet_client, gravity, timestep, frame_skip, world_size: tuple, n_food: int, n_poison: int):
        """
        Creates an gather scene with food and poison spawned in on episode restart

        :param world_size: tuple of the x and y dimensions of the world must be less than 50
        """
        super().__init__(bullet_client, gravity, timestep, frame_skip)
        self.loaded = False
        self.multiplayer = False
        self.size = world_size

        self.ground_plane_mjcf = []
        self.food = {}
        self.poison = {}
        self.n_food = n_food
        self.n_poison = n_poison

        self.rs: np.random.RandomState = np.random.RandomState()

    def seed(self, rs: np.random.RandomState):
        self.rs = rs

    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)
        if not self.loaded:
            self.loaded = True
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'plane.xml'), (0, 0, 0))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'), (0, self.size[1] / 2, 2.5))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'), (0, -self.size[1] / 2, 2.5))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'),
                                                        (self.size[0] / 2, 0, 2.5),
                                                        pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)))]
            self.ground_plane_mjcf += [self._p.loadURDF(join(assets_dir, 'wall.xml'),
                                                        (-self.size[0] / 2, 0, 2.5),
                                                        pybullet.getQuaternionFromEuler((0, 0, np.pi / 2)))]

            for i in self.ground_plane_mjcf:
                self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
                self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, i)

            # TODO reloading this is probably a good idea, but it can take a while
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

