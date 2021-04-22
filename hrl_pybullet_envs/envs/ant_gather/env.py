import math
import warnings
from typing import Dict, Tuple

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
                 n_bins=5,
                 sensor_range=20.,
                 sensor_span=np.pi,
                 robot_coll_dist=1,  # this might be too easy
                 robot_object_spacing=2.,
                 dying_cost=-10,
                 use_sensor=True,
                 respawn=True,
                 render=False,
                 debug=False):
        super().__init__()
        self._alive = True
        self.walk_target_x = 0
        self.walk_target_y = 0

        # env related
        self.n_bins = n_bins
        self.sensor_span = sensor_span
        self.sensor_range = sensor_range
        self.use_sensor = use_sensor
        self.dying_cost = dying_cost
        self.robot_coll_dist = robot_coll_dist

        self.debug = debug

        # scene related
        self.stadium_scene: GatherScene = None
        self.n_food = n_food
        self.n_poison = n_poison
        self.world_size = world_size
        self.spacing = robot_object_spacing
        self.respawn = respawn

        self.observation_space = self.robot.observation_space
        n_food_obs = 2 * n_bins if self.use_sensor else 2 * (min(n_food, n_bins) + min(n_poison, n_bins))
        self.observation_space.shape = (self.robot.observation_space.shape[0] + n_food_obs,)

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = GatherScene(bullet_client, 9.8, 0.0165 / 4, 4,
                                         self.world_size, self.n_food, self.n_poison, self.spacing, self.respawn)

        return self.stadium_scene

    def seed(self, seed=None):
        super().seed(seed)
        if hasattr(self, 'stadium_scene'):
            self.stadium_scene.seed(seed)

    def reset(self):
        r = super().reset()

        dists = {obj_id: self.sq_dist_robot(pos) for obj_id, pos in self.stadium_scene.all_items.items()}
        fr, pr = self.get_food_obs(dists)
        return np.concatenate([r, fr, pr])

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        # could do a more efficient method, but with around 20 items I wonder if it's worth it?
        dists = {obj_id: self.sq_dist_robot(pos) for obj_id, pos in self.stadium_scene.all_items.items()}

        food_reward = 0
        pos = self.parts['torso'].get_position()
        if self.robot_coll_dist > 0:  # otherwise use pybullet collisions (this is easier for the agent)
            for obj_id, dist in dists.items():
                if dist < self.robot_coll_dist:
                    food_reward += self.stadium_scene.reward_collision(obj_id, pos)
                    dists[obj_id] = self.sq_dist_robot(self.stadium_scene.all_items[obj_id])  # updating dist

        # removing angle to target and adding in yaw and food readings
        fr, pr = self.get_food_obs(dists)
        state = np.concatenate([state, fr, pr])


        # state[0] is body height above ground, body_rpy[1] is pitch
        self._alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))
        done = self._alive < 0
        if not np.isfinite(state).all():
            warnings.warn(f'~INF~ {state}')
            done = True

        # for i, f in enumerate(self.robot.feet):
        #     contact_ids = set((x[2], x[4]) for x in f.contact_list())
        #     if self.ground_ids & contact_ids:
        #         # see Issue 63: https://github.com/openai/roboschool/issues/63
        #         self.robot.feet_contact[i] = 1.0
        #     else:
        #         self.robot.feet_contact[i] = 0.0

        if self.robot_coll_dist <= 0:  # otherwise use distance based collision (this is harder)
            collided_ids = [cp[2] for cp in self._p.getContactPoints(self.robot.objects[0])]
            for coll_id in collided_ids:
                food_reward += self.stadium_scene.reward_collision(coll_id, pos)

        dead_rew = self.dying_cost if self._alive < 0 else 0
        return state, food_reward + dead_rew, bool(done), {'food_rew': food_reward, 'dead_rew': dead_rew}

    def get_food_obs(self, dists: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_sensor:
            return self.get_sensor_readings(dists)
        else:
            return self.get_abs_pos(dists)

    # modified from: rllab.envs.mujoco.gather.gather_env.py
    def get_sensor_readings(self, dists: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
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

            if typ == self.FOOD:
                food_readings[bin_number] = intensity
            elif typ == self.POISON:
                poison_readings[bin_number] = intensity
            else:
                raise Exception(f'Unknown food type: {typ}')

            # useful debug
            if self.debug and intensity > 0:
                colour = [1, 0, 0] if typ == self.POISON else [0, 1, 0]
                self._p.addUserDebugLine(self.robot_body.pose().xyz(), [ox, oy, 0], lifeTime=0.5, lineColorRGB=colour)
                self._p.addUserDebugText(f'{intensity:0.2f}', [ox, oy, 0], lifeTime=0.5, textColorRGB=colour)

        return food_readings, poison_readings

    def get_abs_pos(self, dists: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray]:
        food = [pos[:2] + [dists[obj_id]] for obj_id, pos in self.stadium_scene.food.items()]
        poison = [pos[:2] + [dists[obj_id]] for obj_id, pos in self.stadium_scene.poison.items()]

        sorted_food = sorted(food, key=lambda o: o[2])
        sorted_poison = sorted(poison, key=lambda o: o[2])

        if self.debug:
            robot_pos = self.robot_body.pose().xyz()
            for f in sorted_food[:self.n_bins]:
                self._p.addUserDebugLine(robot_pos, [*f[:2], 0], lifeTime=0.5, lineColorRGB=[0, 1, 0])
                self._p.addUserDebugText(f'{f[0]:0.2f},{f[1]:0.2f}', [*f[:2], 0], lifeTime=0.5, textColorRGB=[0, 1, 0])
            for p in sorted_poison[:self.n_bins]:
                self._p.addUserDebugLine(robot_pos, [*p[:2], 0], lifeTime=0.5, lineColorRGB=[1, 0, 0])
                self._p.addUserDebugText(f'{p[0]:0.2f},{p[1]:0.2f}', [*p[:2], 0], lifeTime=0.5, textColorRGB=[1, 0, 0])

        return np.array([f[:2] for f in sorted_food[:self.n_bins]]).flatten(), \
               np.array([p[:2] for p in sorted_poison[:self.n_bins]]).flatten()

    def sq_dist_robot(self, pos):
        xyz = self.parts['torso'].get_pose()
        return (pos[0] - xyz[0]) ** 2 + (pos[1] - xyz[1]) ** 2
