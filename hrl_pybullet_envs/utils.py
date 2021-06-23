# taken from pybullet-gym

import inspect
import os
from enum import Enum

from pybullet_envs.robot_bases import BodyPart

from hrl_pybullet_envs.assets import assets_dir

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data


def get_player_cube(p, x, y, z):
    objects = p.loadMJCF(os.path.join(assets_dir, "player_cube.xml"))
    sphere = objects[0]
    p.resetBasePositionAndOrientation(sphere, [x, y, z], [0, 0, 0, 1])
    p.changeDynamics(sphere, -1, linearDamping=0.9)
    p.changeVisualShape(sphere, -1, rgbaColor=[0, 0.2, 0.8, 0.75])

    part_name, _ = p.getBodyInfo(sphere)
    part_name = part_name.decode("utf8")
    bodies = [sphere]

    p.applyExternalForce(sphere, -1, [10, 10, 10], [x, y, z], flags=p.WORLD_FRAME)

    return BodyPart(p, part_name, bodies, 0, -1)


def get_cube(p, x, y, z):
    body = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube.urdf"), [x, y, z])
    p.changeDynamics(body, -1, mass=0.4)  # match Roboschool
    part_name, _ = p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    p.changeVisualShape(body, -1, rgbaColor=[0, 0.2, 0.8, 0.75])
    return BodyPart(p, part_name, bodies, 0, -1)


def get_sphere(p, x, y, z):
    body = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "sphere2red_nocol.urdf"), [x, y, z])
    part_name, _ = p.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    p.changeVisualShape(body, -1, rgbaColor=[0, 0.2, 0.8, 0.75])
    return BodyPart(p, part_name, bodies, 0, -1)


def debug_draw_point(bullet_client, x, y, colour=None):
    if colour is None:
        colour = [0, 0, 0]

    bullet_client.addUserDebugLine([x + 0.5, y + 0.5, 0],
                                   [x - 0.5, y - 0.5, 0],
                                   lifeTime=0.5,
                                   lineColorRGB=colour)
    bullet_client.addUserDebugLine([x + 0.5, y - 0.5, 0],
                                   [x - 0.5, y + 0.5, 0],
                                   lifeTime=0.5,
                                   lineColorRGB=colour)


class PositionEncoding(Enum):
    normed_vec = 0  # normalized vector from robot position to target
    angle = 1  # sin and cos of angle to target
    # TODO add target to sensor?
