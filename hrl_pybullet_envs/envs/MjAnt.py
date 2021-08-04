# modified from: pybulletgym/envs/mujoco/robots/locomotors/ant.py

import numpy as np
from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.mujoco.robots.robot_bases import MJCFBasedRobot
from pybulletgym.envs.mujoco.robots.locomotors.ant import Ant


class MjAnt(Ant):
    """Removing extra obs and adding positional obs"""

    def __init__(self):
        WalkerBase.__init__(self, power=2.5)
        MJCFBasedRobot.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=29)

    def calc_state(self):
        WalkerBase.calc_state(self)
        pose = self.parts['torso'].get_pose()
        qpos = np.hstack((pose, [j.get_position() for j in self.ordered_joints])).flatten()  # shape (15,)

        velocity = self.parts['torso'].get_velocity()
        qvel = np.hstack((velocity[0], velocity[1], [j.get_velocity() for j in self.ordered_joints])).flatten()

        return np.concatenate([qpos.flat, qvel.flat])

    def alive_bonus(self, z, pitch):
        return +1 if z - self.initial_z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class AntMjEnv(WalkerBaseMuJoCoEnv):
    def __init__(self):
        self.robot = MjAnt()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        # only change from pybulletgym is state[0] -> state[2] for custom ant env
        # state[2]! is body height above ground, body_rpy[1] is pitch
        alive = float(self.robot.alive_bonus(state[2] + self.robot.initial_z, self.robot.body_rpy[1]))
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            # print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if self.ground_ids & contact_ids:
                # see Issue 63: https://github.com/openai/roboschool/issues/63
                # feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        # electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        debugmode = 0
        if debugmode:
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            # print("electricity_cost")
            # print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        self.rewards = [
            alive,
            progress,
            # electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}
