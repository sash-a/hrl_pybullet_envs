from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.ant_gather.scene import GatherScene


class AntGatherBulletEnv(AntBulletEnv):
    def __init__(self, render=False):
        super().__init__(render=render)
        self.walk_target_x = 0
        self.walk_target_y = -15
        self.stadium_scene: GatherScene = None

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = GatherScene(bullet_client, 9.8, 0.0165 / 4, 4, 10, 10, True)
        return self.stadium_scene

    def step(self, a):
        super(AntGatherBulletEnv, self).step(a)
        collided_ids = [cp[2] for cp in self._p.getContactPoints(self.robot.objects[0])]
        # print(collided_ids)
        # print(f'contact points:{self._p.getContactPoints(self.robot.objects[0])}')
        # print(self.robot_body)
        # print(self.ground_ids)
        # print(self.robot.objects)
        for coll_id in collided_ids:
            if coll_id not in self.stadium_scene.ground_plane_mjcf and coll_id != self.robot.objects[0]:
                self.stadium_scene.destroy_and_respawn(coll_id)
