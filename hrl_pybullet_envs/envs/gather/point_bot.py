import os

import numpy as np
import pybullet
from pybullet_envs.robot_bases import MJCFBasedRobot, BodyPart, Pose_Helper

from hrl_pybullet_envs.assets import assets_dir


class PointBot(MJCFBasedRobot):
    no_rot = pybullet.getQuaternionFromEuler((0, 0, 0))
    start_pos = [0, 0, 0.5]

    def __init__(self):
        act_dim = 2
        obs_dim = 8
        super().__init__(os.path.join(assets_dir, "player_cube.xml"), "sphere", act_dim, obs_dim)
        self.initial_z = 1
        self.walk_target_x = 0
        self.walk_target_y = 0

    def reset_pose(self, position, orientation):
        self._p.resetBasePositionAndOrientation(self.objects[0], position, orientation)

    def robot_specific_reset(self, bullet_client):
        self.reset_pose(PointBot.start_pos, PointBot.no_rot)

    def apply_action(self, a):
        a = a / np.linalg.norm(a) * 500
        pos, ori = self._p.getBasePositionAndOrientation(self.objects[0])
        self._p.applyExternalForce(self.objects[0], -1, [*a, 0], pos, self._p.WORLD_FRAME)

    def addToScene(self, bullet_client, bodies):
        assert len(bodies) == 1, f'expected 1 body, got {len(bodies)}'
        self._p = bullet_client

        body = bodies[0]
        part_name, body_name = bullet_client.getBodyInfo(body)
        part_name = part_name.decode("utf8")
        bodies = [body]

        root = BodyPart(self._p, part_name, bodies, 0, -1)
        parts = {part_name: root}
        self.robot_body = root

        return parts, [], [], self.robot_body  # TODO sort out joints

    def calc_state(self):
        # return np.concatenate((pos, vel, ang_vel))
        pose: Pose_Helper = self.robot_body.pose()
        x, y, z = pose.xyz()
        r, p, yaw = pose.rpy()
        walk_target_theta = np.arctan2(self.walk_target_y - y, self.walk_target_x - x)
        # walk_target_dist = np.linalg.norm([self.walk_target_y - pose.xyz()[1], self.walk_target_x - pose.xyz()[0]])
        angle_to_target = walk_target_theta - yaw

        rot_speed = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [0, 0, 1]]
        )
        vx, vy, vz = np.dot(rot_speed, self.robot_body.speed())  # rotate speed back to body point of view

        return np.array([z - self.initial_z,  # this never changes, but its consistent with other walkers obs
                         np.sin(angle_to_target), np.cos(angle_to_target),
                         0.3 * vx, 0.3 * vy, 0.3 * vz,  # 0.3 is just scaling typical speed into -1..+1
                         r, p], dtype=np.float32)

    def calc_potential(self):
        # TODO...?
        pass

    def alive_bonus(self, z, pitch):
        return 1  # point bot can't die
