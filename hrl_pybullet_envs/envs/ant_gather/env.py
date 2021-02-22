import math
import warnings

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
                 robot_object_spacing=2.,
                 dying_cost=-10,
                 render=False):
        super().__init__(render=render)
        self._alive = True
        self.walk_target_x = 0
        self.walk_target_y = 0

        self.n_bins = n_bins
        self.sensor_span = sensor_span
        self.sensor_range = sensor_range
        self.spacing = robot_object_spacing
        self.dying_cost = dying_cost

        self.stadium_scene: GatherScene = None
        self.n_food = n_food
        self.n_poison = n_poison
        self.world_size = world_size

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = GatherScene(bullet_client, 9.8, 0.0165 / 4, 4, self.world_size, self.n_food, self.n_poison)
        self.stadium_scene.seed(self.np_random)
        return self.stadium_scene

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit
        # removing angle to target and adding in yaw and food readings
        fr, pr = self.get_readings()
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

        food_reward = 0
        collided_ids = [cp[2] for cp in self._p.getContactPoints(self.robot.objects[0])]
        for coll_id in collided_ids:
            food_reward += self.stadium_scene.destroy_and_respawn(coll_id)

        dead_rew = self.dying_cost if self._isDone() else 0

        return state, food_reward + dead_rew, bool(done), {'food_rew': food_reward, 'dead_rew': dead_rew}

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

        # TODO:
        #  calc dist once
        #  possibly do reward based on dist to food/poison rather than collision with it
        sorted_objects = sorted(
            objects, key=lambda o:
            (o[0] - robot_x) ** 2 + (o[1] - robot_y) ** 2)[::-1]
        # fill the readings
        bin_res = self.sensor_span / self.n_bins

        ori = self.robot_body.pose().rpy()[2]  # main change from rllab, this is the yaw of the robot

        for ox, oy, typ in sorted_objects:
            # TODO second dist calc is here
            dist = ((oy - robot_y) ** 2 + (ox - robot_x) ** 2) ** 0.5  # compute distance between object and robot
            if dist > self.sensor_range:  # only include readings for objects within range
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
