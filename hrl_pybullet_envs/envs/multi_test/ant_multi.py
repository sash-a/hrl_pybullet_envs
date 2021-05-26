import pybullet
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.gym_locomotion_envs import AntBulletEnv
import numpy as np
from hrl_pybullet_envs.envs.sizeable_enclosed_scene import VecSquareEnclosedScene
from pybullet_envs.robot_locomotors import Ant


class CustomAnt(Ant):
    def __init__(self, center, n):
        super().__init__()

        self.start_pos_x, self.start_pos_y, self.start_pos_z = center[0], center[1], center[2]
        self.center = center
        self.player_n = n

        self.np_random = np.random.RandomState()

    def addToScene(self, bullet_client, bodies):
        r = super().addToScene(bullet_client, bodies)
        self.robot_body.reset_position(self.center)
        return r

    def robot_specific_reset(self, bullet_client):
        self._p = bullet_client
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)
        # self.robot_body.reset_position(self.center)

        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None


class AntVecEnv(AntBulletEnv):
    def __init__(self, n):
        super().__init__()
        self.n = n
        centers = VecSquareEnclosedScene.calc_centers(n, 15)

        self.robots = [CustomAnt([*c, 0.55], i) for i, c in enumerate(centers)]
        self.robot = self.robots[0]

        self.all_parts = []
        self.all_jdicts = []
        self.all_ordered_joints = []

    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = VecSquareEnclosedScene(bullet_client, 9.8, 0.0165 / 4, 4, 15, self.n)
        return self.stadium_scene

    def reset(self):
        if self.stateId >= 0:
            self._p.restoreState(self.stateId)

        rs = []
        for robot in self.robots:
            self.robot = robot
            rs += [MJCFBaseBulletEnv.reset(self)]
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

            parts, jdict, ordered_joints, robot_body = robot.addToScene(self._p, self.stadium_scene.ground_plane_mjcf)
            self.all_parts += [parts]
            self.all_jdicts += [jdict]
            self.all_ordered_joints += [ordered_joints]

        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        self.robot_body = self.robots[0].robot_body
        self.ground_ids = set([(parts[f].bodies[parts[f].bodyIndex], parts[f].bodyPartIndex)
                               for f in self.foot_ground_object_names])

        if self.stateId < 0: self.stateId = self._p.saveState()
        return rs

    def step(self, a):
        steps = []

        for i in range(self.n):
            act = a[i]
            self.robot = self.robots[i]
            # I don't think this is necessary, but adding for continuity anyway
            self.parts = self.all_parts[i]
            self.jdict = self.all_jdicts[i]
            self.ordered_joints = self.all_ordered_joints[i]

            steps += [super().step(act)]

        return steps
