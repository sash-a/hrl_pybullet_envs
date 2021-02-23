import math
import warnings
from typing import Dict

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.envs.ant_gather.scene import GatherScene


class AntGatherBulletEnv(AntBulletEnv):
    FOOD = 'food'
    POISON = 'poison'

    def __init__(self,
                 n_food=8,
                 n_poison=8,
                 world_size=(15, 15),
                 n_bins=10,
                 sensor_range=8.,
                 sensor_span=np.pi,
                 robot_coll_dist=1,  # this might be too easy
                 robot_object_spacing=2.,
                 dying_cost=-10,
                 respawn=True,
                 render=False):
        super().__init__(render=render)
        self._alive = True
        self.walk_target_x = 0
        self.walk_target_y = 0

        # env related
        self.n_bins = n_bins
        self.sensor_span = sensor_span
        self.sensor_range = sensor_range
        self.dying_cost = dying_cost
        self.robot_coll_dist = robot_coll_dist

        # scene related
        self.stadium_scene: GatherScene = None
        self.n_food = n_food
        self.n_poison = n_poison
        self.world_size = world_size
        self.spacing = robot_object_spacing
        self.respawn = respawn

        # removing angle to target and adding in yaw and food readings
        self.observation_space = self.robot.observation_space
        self.observation_space.shape = (self.observation_space.shape[0] - 2 + 1 + 2 * n_bins,)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = GatherScene(bullet_client, 9.8, 0.0165 / 4, 4,
                                         self.world_size, self.n_food, self.n_poison, self.spacing, self.respawn)
        self.stadium_scene.seed(self.np_random)
        return self.stadium_scene

    def reset(self):
        r = super().reset()

        dists = {obj_id: self.sq_dist_robot(pos) for obj_id, pos in self.stadium_scene.all_items.items()}
        fr, pr = self.get_readings(dists)
        return np.concatenate([r[:1], r[3:8], [self.robot.body_rpy[2]], r[8:], fr, pr])

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        # could do a more efficient method, but with around 20 items I wonder if it's worth it?
        dists = {obj_id: self.sq_dist_robot(pos) for obj_id, pos in self.stadium_scene.all_items.items()}

        food_reward = 0
        if self.robot_coll_dist > 0:  # otherwise use pybullet collisions (this is easier for the agent)
            for obj_id, dist in dists.items():
                if dist < self.robot_coll_dist:
                    food_reward += self.stadium_scene.reward_collision(obj_id, self.robot.body_real_xyz)
                    dists[obj_id] = self.sq_dist_robot(self.stadium_scene.all_items[obj_id])  # updating dist

        # removing angle to target and adding in yaw and food readings
        fr, pr = self.get_readings(dists)
        state = np.concatenate([state[:1], state[3:8], [self.robot.body_rpy[2]], state[8:], fr, pr])

        # state[0] is body height above ground, body_rpy[1] is pitch
        self._alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))
        done = self._isDone()
        if not np.isfinite(state).all():
            warnings.warn(f'~INF~ {state}')
            done = True

        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        if self.robot_coll_dist <= 0:  # otherwise use distance based collision (this is harder)
            collided_ids = [cp[2] for cp in self._p.getContactPoints(self.robot.objects[0])]
            for coll_id in collided_ids:
                food_reward += self.stadium_scene.reward_collision(coll_id, self.robot.body_real_xyz)

        dead_rew = self.dying_cost if self._isDone() else 0
        return state, food_reward + dead_rew, bool(done), {'food_rew': food_reward, 'dead_rew': dead_rew}

    # modified from: rllab.envs.mujoco.gather.gather_env.py
    def get_readings(self, dists: Dict[int, float]):
        """compute sensor readings"""
        # first, obtain current orientation
        food_readings = np.zeros(self.n_bins)
        poison_readings = np.zeros(self.n_bins)
        robot_x, robot_y = self.robot_body.pose().xyz()[:2]
        ori = self.robot_body.pose().rpy()[2]  # main change from rllab, this is the yaw of the robot

        # sort objects by distance to the robot, so that farther objects'
        # signals will be occluded by the closer ones'
        objects = [pos[:2] + [self.FOOD] + [dists[obj_id]] for obj_id, pos in self.stadium_scene.food.items()] + \
                  [pos[:2] + [self.POISON] + [dists[obj_id]] for obj_id, pos in self.stadium_scene.poison.items()]

        sorted_objects = sorted(objects, key=lambda o: o[3], reverse=True)
        bin_res = self.sensor_span / self.n_bins

        for ox, oy, typ, dist in sorted_objects:
            if dist > self.sensor_range:  # only include readings for objects within range
                continue

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

            # useful debug
            # self._p.addUserDebugLine(self.robot_body.pose().xyz(), [ox, oy, 0], lifeTime=0.1,
            #                          lineColorRGB=[1, 0, 0] if typ == self.POISON else [0, 1, 0])
            if typ == self.FOOD:
                food_readings[bin_number] = intensity
            elif typ == self.POISON:
                poison_readings[bin_number] = intensity
            else:
                raise Exception(f'Unknown food type: {typ}')

        return food_readings, poison_readings

    def sq_dist_robot(self, pos):
        return (pos[0] - self.robot.body_real_xyz[0]) ** 2 + (pos[1] - self.robot.body_real_xyz[1]) ** 2
