import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

from hrl_pybullet_envs.envs.ant_maze.scene import MazeScene


class AntMazeBulletEnv(AntBulletEnv):
    """
    At evaluation time, we evaluate the agent only on its ability to reach (0,16). We define a “success” as being within
    an L2 distance of 5 from the target on the ultimate step of the episode. - Data efficient HRL
    """

    def __init__(self, render=False):
        super().__init__(render=render)
        self.walk_target_x = 0
        self.walk_target_y = -15

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = MazeScene(bullet_client, 9.8, 0.0165 / 4, 4, (20, 20))
        return self.stadium_scene

    def step(self, a):
        pos_before = self.robot_body.pose().xyz()
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit
        pos_after = self.robot_body.pose().xyz()

        # state[0] is body height above ground, body_rpy[1] is pitch
        self._alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        # feet_collision_cost = 0.0
        # TODO: Maybe calculating feet contacts could be done within the robot code
        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        #
        # joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        # debugmode = 0
        # if (debugmode):
        #     print("alive=")
        #     print(self._alive)
        #     print("progress")
        #     print(progress)
        #     print("electricity_cost")
        #     print(electricity_cost)
        #     print("joints_at_limit_cost")
        #     print(joints_at_limit_cost)
        #     print("feet_collision_cost")
        #     print(feet_collision_cost)

        # self.rewards = [self._alive, electricity_cost, joints_at_limit_cost, feet_collision_cost]

        # if (debugmode):
        #     print("rewards=")
        #     print(self.rewards)
        #     print("sum rewards")
        #     print(sum(self.rewards))
        self.HUD(state, a, done)
        # self.reward += sum(self.rewards)

        # mujoco ant rew:
        # TODO not sure what to put here: https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/envs/ant.py
        #  seems to reward simply moving forward in x dir, which makes no sense. Paper implies that it gives reward
        #  *only* for ending up 5 away from the target

        np_walk_target = np.array([self.walk_target_x, self.walk_target_y])
        dist_before = np.linalg.norm(np.array(pos_before[:2] - np_walk_target))
        dist_after = np.linalg.norm(np.array(pos_after[:2] - np_walk_target))

        forward_reward = dist_before - dist_after  # / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        survive_reward = 1.0
        rewards = [forward_reward - ctrl_cost + survive_reward]
        if -5 < dist_after < 5:
            rewards += 1000

        return state, sum(rewards), bool(done), {'dist':        dist_after,
                                                 'forward_rew': forward_reward,
                                                 'ctrl_cost':   ctrl_cost}

    def reset(self):
        return super().reset()
