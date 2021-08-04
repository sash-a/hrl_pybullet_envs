from os.path import join

from hrl_pybullet_envs.assets import assets_dir
from hrl_pybullet_envs.envs.sizeable_enclosed_scene import SizeableEnclosedScene
from hrl_pybullet_envs.envs.intersection_utils import Point


class MazeScene(SizeableEnclosedScene):
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        world_size = (8, 14)
        super().__init__(bullet_client, gravity, timestep, frame_skip, world_size, (0, 0))
        self.box_size = (4, 4)  # this doesn't affect the size, this is simply copying from the asset
        self.box_pos = (-(world_size[0] - self.box_size[0]) / 2, 0)

        box_p1 = Point(self.box_pos[0] + self.box_size[0] / 2, self.box_pos[1] + self.box_size[1] / 2)
        box_p2 = Point(self.box_pos[0] - self.box_size[0] / 2, self.box_pos[1] - self.box_size[1] / 2)

        # (box_p1, Point(box_p2.x, box_p1.y)) is the top box in line with the wall
        self.box_bounds = [(box_p1, Point(box_p1.x, box_p2.y)),
                           (box_p2, Point(box_p2.x, box_p1.y)),
                           (box_p2, Point(box_p1.x, box_p2.y))]

        # world_p1 = Point(world_size[0] / 2, world_size[1] / 2)
        # world_p2 = Point(-world_size[0] / 2, -world_size[1] / 2)
        #
        # self.world_bounds = [(world_p1, Point(world_p2.x, world_p1.y)),
        #                      (world_p1, Point(world_p1.x, world_p2.y)),
        #                      (world_p2, Point(world_p2.x, world_p1.y)),
        #                      (world_p2, Point(world_p1.x, world_p2.y))]

    bounds = property(lambda self: self.world_bounds + self.box_bounds)

    def episode_restart(self, bullet_client):
        if not self.loaded:
            box = bullet_client.loadURDF(join(assets_dir, 'box.xml'), basePosition=(*self.box_pos, 1))
            self.ground_plane_mjcf += [box]

        super().episode_restart(bullet_client)
