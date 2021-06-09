# modified from: pybulletgym/envs/mujoco/robots/locomotors/ant.py

import numpy as np
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv, AntBulletEnv
from pybullet_envs.robot_locomotors import Ant, WalkerBase


class MjAnt(Ant):
    def __init__(self, action_dim=8, obs_dim=23):
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=action_dim, obs_dim=obs_dim, power=2.5)

    def calc_state(self):
        super().calc_state()
        pose = self.parts['torso'].get_pose()
        qpos = np.hstack((pose, [j.get_position() for j in self.ordered_joints])).flatten()

        velocity = self.parts['torso'].speed()
        qvel = np.hstack((velocity[0], velocity[1], [j.get_velocity() for j in self.ordered_joints])).flatten()

        return np.concatenate([
            qpos.flat[2:],  # self.sim.data.qpos.flat[2:],
            qvel.flat,  # self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def alive_bonus(self, z, pitch):
        return +1 if z - self.initial_z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


class AntMjEnv(WalkerBaseBulletEnv):
    def __init__(self):
        self.robot = MjAnt(obs_dim=25)
        super().__init__(self.robot)

    progress_weight = 1
    ctrl_cost_weight = 1
    survive_reward_weight = 1

    def step(self, a):
        if not self.scene.multiplayer:
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        # state[0] is body height above ground, body_rpy[1] is pitch
        self._alive = float(self.robot.alive_bonus(state[0] + self.robot.initial_z, self.robot.body_rpy[1]))
        done = self._isDone()
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()

        progress = self.progress_weight * float(self.potential - potential_old)
        ctrl_cost = self.ctrl_cost_weight * .5 * np.square(a).sum()
        survive_reward = self.survive_reward_weight * 1.0
        reward = progress - ctrl_cost + survive_reward

        self.update_foot_contacts()

        return state, reward, bool(done), {'ctrl_cost': ctrl_cost, 'survive_reward': survive_reward,
                                           'progress':  progress}

    def reset(self):
        if hasattr(self, '_p'):
            self._p.resetBasePositionAndOrientation(self.robot.objects[0], [0, 0, 1], [0, 0, 0, 1])
        return super().reset()

    def update_foot_contacts(self):
        for i, f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            if self.ground_ids & contact_ids:
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
