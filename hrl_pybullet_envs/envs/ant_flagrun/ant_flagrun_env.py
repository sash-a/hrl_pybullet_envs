from typing import Tuple, List

import numpy as np
from gym.spaces import Box
from pybullet_envs.gym_locomotion_envs import AntBulletEnv, WalkerBaseBulletEnv

from hrl_pybullet_envs.envs.sizeable_enclosed_scene import SizeableEnclosedScene
from hrl_pybullet_envs.utils import get_sphere


class AntFlagrunBulletEnv(AntBulletEnv):
    """Useful env for pretraining the ant for the other envs"""

    def __init__(self, size=10, tolerance=0.5, max_targets=100, max_target_dist=0, timeout=200, enclosed=True,
                 use_sensor=False, sensor_bins=8, sensor_span=np.pi, sensor_range=4,
                 switch_flag_on_collision=True, manual_goal_creation=False, seed=123, debug=False):
        assert (max_target_dist == 0 and max_targets > 0) or (max_targets <= 0 and max_target_dist > 0), \
            'cannot have both max_targets and max_target_dist set at the same time'
        super().__init__()
        self.size = size
        self.tol = tolerance
        self.max_targets = max_targets
        self.max_target_dist = max_target_dist
        self.timeout = timeout
        self.enclosed = enclosed
        self.switch_flag_on_collision = switch_flag_on_collision
        self.manual_goal_creation = manual_goal_creation
        self.debug = debug

        # sensors
        self.use_sensor = use_sensor
        self.n_bins = sensor_bins
        self.sensor_span = sensor_span
        self.sensor_range = sensor_range

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

        self._rewarded = False  # true if rewarded for reaching current target

        if use_sensor:
            super_obs = self.robot.observation_space
            self.observation_space = Box(-np.inf, np.inf, (super_obs.shape[0] + sensor_bins,))

    goal = property(lambda self: (self.walk_target_x, self.walk_target_y))

    def create_single_player_scene(self, bullet_client):
        if self.enclosed or self.use_sensor:
            # if enclosed make the enclosing area larger so that no targets spawn on/past the wall
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

        return g

    def create_close_target(self) -> Tuple[float, float]:
        world_bound = self.size / 2
        g = (world_bound + 1, world_bound + 1)
        r = self.mpi_common_rand
        while not (-world_bound < g[0] < world_bound and -world_bound < g[1] < world_bound):
            g = (r.uniform(self.tol, self.max_target_dist / 2) * (r.randint(0, 2) * 2 - 1),
                 r.uniform(self.tol, self.max_target_dist / 2) * (r.randint(0, 2) * 2 - 1))
            g += self.robot.body_real_xyz[:2]

        return g

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

        if self.isRender and self.flag is not None:
            self._p.resetBasePositionAndOrientation(self.flag.bodies[0],
                                                    [self.walk_target_x, self.walk_target_y, 0.7],
                                                    [0, 0, 0, 1])

    def next_target(self):
        if self.max_targets < 1:
            self.set_target(*self.create_close_target())
        else:
            self.set_target(*self.goals.pop())

        self._rewarded = False
        self.potential = self.robot.calc_potential()  # avoid reward jump
        return self.robot.calc_state()

    def _get_obs(self, env_obs):
        if self.use_sensor:
            env_obs = np.concatenate((env_obs, self.scene.sense_walls(self.n_bins,
                                                                      self.sensor_span,
                                                                      self.sensor_range,
                                                                      self.robot.body_real_xyz[:2],
                                                                      self.robot_body.pose().rpy()[2],
                                                                      self.debug)))
        return env_obs

    def reset(self):
        WalkerBaseBulletEnv.electricity_cost = 0  # -2.0
        WalkerBaseBulletEnv.stall_torque_cost = 0  # -0.1
        WalkerBaseBulletEnv.joints_at_limit_cost = 0

        self._rewarded = False
        # AntMjEnv.ctrl_cost_weight = 0
        # AntMjEnv.survive_reward_weight = 0

        super().reset()
        self._p.resetBasePositionAndOrientation(self.robot.objects[0], [0, 0, 0.25], [0, 0, 0, 1])
        self.robot.robot_specific_reset(self.scene._p)
        s = self.robot.calc_state()

        # state modifications
        # rel_dir_to_goal = -np.array(self.goal)
        # s = np.concatenate((rel_dir_to_goal, s))

        self.goals.clear()
        if not self.manual_goal_creation:  # creating a new goal on every reset if not manually creating the goal
            self.create_targets(self.max_targets)
            s = self.next_target()

        return self._get_obs(s)

    ant_env_rew_weight = 1
    path_rew_weight = 0
    dist_rew_weight = 0
    goal_reach_rew = 5000

    def step(self, a):
        s, r, d, i = super().step(a)
        # state modifications: adding in vector towards goal
        # rel_dir_to_goal = np.array(self.goal) - self.robot.body_real_xyz[:2]
        # rel_dir_to_goal = rel_dir_to_goal / np.linalg.norm(rel_dir_to_goal)
        # s = np.concatenate((rel_dir_to_goal, s))

        # reward modifications
        r *= AntFlagrunBulletEnv.ant_env_rew_weight
        self.steps_since_goal_change += 1

        # rewarding agent based on how well it is following a straight line to the goal
        path_rew = np.dot(self.robot.body_real_xyz[:2] - self._goal_start_pos,
                          np.array(self.goal) - self._goal_start_pos) / self._sq_dist_goal
        r += path_rew * AntFlagrunBulletEnv.path_rew_weight

        r += -self.robot.walk_target_dist * AntFlagrunBulletEnv.dist_rew_weight

        if self.debug:
            self.scene._p.addUserDebugLine(self.robot.body_real_xyz, [*self.goal, 0.5], lifeTime=0.1)
        # If close enough to target then give extra reward and move the target.
        if self.robot.walk_target_dist < self.tol:
            if not self._rewarded:
                r += AntFlagrunBulletEnv.goal_reach_rew
                self._rewarded = True

            try:
                if self.switch_flag_on_collision:
                    s = self.next_target()
                    i['target'] = self.goal
                    self.steps_since_goal_change = 0
            except IndexError:
                d = True

        if 0 < self.timeout <= self.steps_since_goal_change:
            try:
                s = self.next_target()
                i['target'] = self.goal
                self.steps_since_goal_change = 0
            except IndexError:
                d = True

        return self._get_obs(s), r, d, i
