from typing import Tuple

import numpy as np
from pybullet_envs.gym_locomotion_envs import AntBulletEnv, WalkerBaseBulletEnv

from hrl_pybullet_envs.envs.MjAnt import AntMjEnv
from hrl_pybullet_envs.envs.sizeable_enclosed_scene import SizeableEnclosedScene
from hrl_pybullet_envs.utils import get_sphere


class AntFlagrunBulletEnv(AntMjEnv):
    """Useful env for pretraining the ant for the other envs"""

    def __init__(self, size=10, tolerance=0.5, max_targets=100, timeout=200, enclosed=False, manual_goal_creation=False,
                 seed=123):
        super().__init__()
        self.size = size
        self.tol = tolerance
        self.max_targets = max_targets
        self.timeout = timeout
        self.enclosed = enclosed
        self.manual_goal_creation = manual_goal_creation

        self.flag = None

        # needed for parallel envs so that they create the same targets on each thread (np_random seems to not work)
        self.mpi_common_rand: np.random.RandomState = np.random.RandomState(seed)
        self.create_target()
        self.flag = None

        self.steps_since_goal_change = 0

        self.goals = []

        self.steps_since_goal_change = 0
        self._sq_dist_goal = 0  # distance to goal on step new goal received
        self._goal_start_pos = np.array([0, 0])  # position on step new goal received

    goal = property(lambda self: (self.walk_target_x, self.walk_target_y))

    def create_single_player_scene(self, bullet_client):
        if self.enclosed:  # if enclosed make the enclosing area larger so that no targets spawn on/past the wall
            scene = self.stadium_scene = SizeableEnclosedScene(bullet_client, 9.8, 0.0165 / 4, 4, (self.size + 2,) * 2)
        else:
            scene = super().create_single_player_scene(bullet_client)

        if self.isRender and not self.flag:
            self.flag = get_sphere(self._p, self.walk_target_x, self.walk_target_y, 0.7)

        return scene

    def create_target(self) -> Tuple[float, float]:
        g = (self.mpi_common_rand.uniform(-self.size / 2, self.size / 2),
             self.mpi_common_rand.uniform(-self.size / 2, self.size / 2))
        while np.linalg.norm(g) < 0.5:  # don't spawn too close too player start (0,0)
            g = (self.mpi_common_rand.uniform(-self.size / 2, self.size / 2),
                 self.mpi_common_rand.uniform(-self.size / 2, self.size / 2))

        return g[0], g[1]

    def create_targets(self, n):
        """
        Used to create multiple goals such that all agents run in parallel will continue to generate the same values.
        Each generation/epoch these should be cleared and recreated
        """
        self.goals = [self.create_target() for _ in range(n)]

    def set_target(self, x, y):
        self.walk_target_x = x
        self.walk_target_y = y
        self.robot.walk_target_x = x
        self.robot.walk_target_y = y

        self._sq_dist_goal = np.linalg.norm(np.array([x, y]) - self.robot.robot_body.get_position()[:2]) ** 2
        self._goal_start_pos = self.robot.robot_body.get_position()[:2]

        if self.isRender:
            self._p.resetBasePositionAndOrientation(self.flag.bodies[0],
                                                    [self.walk_target_x, self.walk_target_y, 0.7],
                                                    [0, 0, 0, 1])

    def reset(self):
        # WalkerBaseBulletEnv.electricity_cost = 0  # -2.0
        # WalkerBaseBulletEnv.stall_torque_cost = 0  # -0.1

        AntMjEnv.ctrl_cost_weight = 0
        AntMjEnv.survive_reward_weight = 0

        s = super().reset()
        # state modifications
        rel_dir_to_goal = self.robot.body_xyz[:2] - np.array(self.goal)
        s = np.concatenate((rel_dir_to_goal, s))

        self.goals.clear()
        if not self.manual_goal_creation:  # creating a new goal on every reset if not manually creating the goal
            self.create_targets(self.max_targets)
            self.set_target(*self.goals.pop())

        return s

    ant_env_rew_weight = 1
    path_rew_weight = 0
    goal_reach_rew = 5000

    def step(self, a):
        s, r, d, i = super().step(a)

        # state modifications: adding in vector towards goal
        rel_dir_to_goal = np.array(self.goal) - self.robot.body_xyz[:2]
        rel_dir_to_goal = rel_dir_to_goal / np.linalg.norm(rel_dir_to_goal)
        s = np.concatenate((rel_dir_to_goal, s))

        # reward modifications
        r *= AntFlagrunBulletEnv.ant_env_rew_weight
        self.steps_since_goal_change += 1

        # dist = np.linalg.norm(self.robot.body_xyz[:2] - np.array(self.goal))
        # rewarding agent based on how well it is following a straight line to the goal
        path_rew = np.dot(self.robot.body_xyz[:2] - self._goal_start_pos,
                          np.array(self.goal) - self._goal_start_pos) / self._sq_dist_goal
        r += path_rew * AntFlagrunBulletEnv.path_rew_weight

        # If close enough to target then give extra reward and move the target.
        if self.robot.walk_target_dist < self.tol:
            r += AntFlagrunBulletEnv.goal_reach_rew
            try:
                self.set_target(*self.goals.pop())
                i['target'] = self.goal
                self.steps_since_goal_change = 0
            except IndexError:
                d = True

        if 0 < self.timeout <= self.steps_since_goal_change:
            try:
                self.set_target(*self.goals.pop())
                i['target'] = self.goal
                self.steps_since_goal_change = 0
            except IndexError:
                d = True

        return s, r, d, i
