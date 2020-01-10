from __future__ import division
import pybullet as p
import pybullet_data
import numpy as np
import pybullet_utils as pu
import rrt
import rrt_connect
import time
import argparse
from ur5 import UR5

UR5_JOINT_INDICES = [1, 2, 3, 4, 5, 6]


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


def draw_sphere_marker(position, radius, color):
    vs_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    marker_id = p.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
    return marker_id


def remove_marker(marker_id):
    p.removeBody(marker_id)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200,
                                 cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")

    initial_pose = [[0, 0, 0.02], [0, 0, 0, 1]]
    initial_joint_values = [-0.813358794499552, -0.37120422397572495, -0.754454729356351, 0, 0, 0]
    ur5_urdf_path = 'assets/ur5/ur5.urdf'
    ur5 = UR5(initial_pose, initial_joint_values, ur5_urdf_path)
    obstacle1 = p.loadURDF('assets/block.urdf',
                           basePosition=[1 / 4, 0, 1 / 2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('assets/block.urdf',
                           basePosition=[2 / 4, 0, 2 / 3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    ## start and goal
    start_conf = [-0.813358794499552, -0.37120422397572495, -0.754454729356351, 0, 0, 0]
    start_position = ur5.forward_kinematics(start_conf)[0]
    goal_conf = (0.7527214782907734, -0.6521867735052328, -0.4949270744967443, 0, 0, 0)
    goal_position = ur5.forward_kinematics(goal_conf)[0]
    goal_marker = draw_sphere_marker(position=goal_position, radius=0.02, color=[1, 0, 0, 1])

    # place hoder to save the solution path
    path_conf = None

    # get the collision checking function
    from collision_utils import get_collision_fn

    collision_fn = get_collision_fn(ur5.id, UR5_JOINT_INDICES, obstacles=obstacles,
                                    attachments=[], self_collisions=True,
                                    disabled_collisions=set())

    if args.birrt:
        if args.smoothing:
            # using birrt with smoothing
            path_conf = rrt_connect.birrt(ur5,
                                          UR5_JOINT_INDICES,
                                          start_conf,
                                          goal_conf,
                                          distance=pu.get_distance_fn(ur5, UR5_JOINT_INDICES),
                                          sample=pu.get_sample_fn(ur5, UR5_JOINT_INDICES),
                                          extend=pu.get_extend_fn(ur5, UR5_JOINT_INDICES),
                                          collision=collision_fn,
                                          smooth=300,
                                          visualize=True)
        else:
            # using birrt without smoothing
            path_conf = rrt_connect.birrt(ur5,
                                          UR5_JOINT_INDICES,
                                          start_conf,
                                          goal_conf,
                                          distance=pu.get_distance_fn(ur5, UR5_JOINT_INDICES),
                                          sample=pu.get_sample_fn(ur5, UR5_JOINT_INDICES),
                                          extend=pu.get_extend_fn(ur5, UR5_JOINT_INDICES),
                                          collision=collision_fn,
                                          smooth=None,
                                          visualize=True)
    else:
        # using rrt
        def goal_test(conf):
            return np.allclose(conf, goal_conf, atol=0.001, rtol=0)


        for i in range(1):
            start_time = time.time()
            path_conf = rrt.rrt(fk=ur5.forward_kinematics,
                                start=start_conf,
                                goal_sample=goal_conf,
                                distance=ur5.arm_distance_fn,
                                sample=ur5.arm_sample_fn,
                                extend=ur5.arm_extend_fn,
                                collision=collision_fn,
                                goal_probability=0.2,
                                iterations=2000,
                                goal_test=goal_test,
                                visualize=True)

            duration = time.time() - start_time
            print(duration)

    if path_conf is None:
        # no collision-free path is found within the time budget
        print("fail")
    else:
        # visualize the path and execute it
        for i in range(len(path_conf)):
            if i != len(path_conf) - 1:
                p1 = ur5.forward_kinematics(path_conf[i])[0]
                p2 = ur5.forward_kinematics(path_conf[i + 1])[0]
                pu.draw_line(p1, p2, rgb_color=[1, 0, 0], width=6)
        while True:
            for q in path_conf:
                ur5.set_arm_joints(q)
                time.sleep(0.5)
