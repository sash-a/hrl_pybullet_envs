from os.path import join

from hrl_pybullet_envs.assets import assets_dir
from hrl_pybullet_envs.envs.sizeable_enclosed_scene import SizeableEnclosedScene


class MazeScene(SizeableEnclosedScene):
    def __init__(self, bullet_client, gravity, timestep, frame_skip, world_size):
        super().__init__(bullet_client, gravity, timestep, frame_skip, world_size, (8, 8))

    def episode_restart(self, bullet_client):
        if not self.loaded:
            self.ground_plane_mjcf += [bullet_client.loadURDF(join(assets_dir, 'box.xml'), basePosition=(8, 5, 2.5))]

        super().episode_restart(bullet_client)
