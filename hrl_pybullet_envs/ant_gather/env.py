import math
import warnings

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.ant_gather.scene import GatherScene


class AntGatherBulletEnv(AntBulletEnv):
    FOOD = 'food'
    POISON = 'poison'

    def __init__(self,
                 n_food=8,
                 n_poison=8,
                 activity_range=6.,
                 robot_object_spacing=2.,
                 catch_range=1.,
                 n_bins=10,
                 sensor_range=8.,
                 sensor_span=np.pi,
                 coef_inner_rew=0.,
                 dying_cost=-10,
                 render=False):
        super().__init__(render=render)
        self.walk_target_x = 0
        self.walk_target_y = -15
        self.stadium_scene: GatherScene = None
        self.n_bins = n_bins
        self.spacing = robot_object_spacing
        self.sensor_span = sensor_span
        self.sensor_range = sensor_range

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = GatherScene(bullet_client, 9.8, 0.0165 / 4, 4, 10, 10, True)
        return self.stadium_scene

    def step(self, a):
        super(AntGatherBulletEnv, self).step(a)
        collided_ids = [cp[2] for cp in self._p.getContactPoints(self.robot.objects[0])]

        for coll_id in collided_ids:
            if coll_id not in self.stadium_scene.ground_plane_mjcf and coll_id != self.robot.objects[0]:
                self.stadium_scene.destroy_and_respawn(coll_id)

        fr, pr = self.get_readings()
        print(f'food:{fr}\npoison:{pr}')

    # modified from: rllab.envs.mujoco.gather.gather_env.py
    def get_readings(self):  # equivalent to get_current_maze_obs in maze_env.py
        # compute sensor readings
        # first, obtain current orientation
        food_readings = np.zeros(self.n_bins)
        poison_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.robot_body.pose().xyz()[:2]
        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        objects = [pos[:2] + [self.FOOD] for pos in self.stadium_scene.food.values()] + \
                  [pos[:2] + [self.POISON] for pos in self.stadium_scene.poison.values()]
        sorted_objects = sorted(
            objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.robot_body.pose().rpy()[2]  # no idea if this is correct - should be the yaw

        for ox, oy, typ in sorted_objects:
            # compute distance between object and robot
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5
            # only include readings for objects within range
            if dist > self.sensor_range:
                continue  # break? because its sorted by distance
            angle = math.atan2(oy - robot_y, ox - robot_x) - ori
            if math.isnan(angle):
                warnings.warn('angle is nan')
            angle = angle % (2 * math.pi)
            if angle > math.pi:
                angle = angle - 2 * math.pi
            if angle < -math.pi:
                angle = angle + 2 * math.pi
            # outside of sensor span - skip this
            half_span = self.sensor_span * 0.5
            if abs(angle) > half_span:
                continue
            bin_number = int((angle + half_span) / bin_res)
            intensity = 1.0 - dist / self.sensor_range

            # self._p.addUserDebugLine(self.robot_body.pose().xyz(), [ox, oy, 0], lifeTime=1)

            if typ == self.FOOD:
                food_readings[bin_number] = intensity
            elif typ == self.POISON:
                poison_readings[bin_number] = intensity
            else:
                raise Exception('Unknown food type')

        return food_readings, poison_readings
