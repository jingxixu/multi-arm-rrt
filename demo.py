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
import json
from ur5_group import UR5Group
import itertools


UR5_JOINT_INDICES = [1, 2, 3, 4, 5, 6]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--birrt', action='store_true', default=False)
    parser.add_argument('--smoothing', action='store_true', default=False)
    parser.add_argument('--world_file', type=str, default='worlds/single.json')
    args = parser.parse_args()

    args.world_config = json.load(open(args.world_file))
    return args


if __name__ == "__main__":
    args = get_args()

    initial_base_positions = [d['initial_base_position'] for d in args.world_config['robots']]
    initial_base_quaternions = [d['initial_base_quaternion'] for d in args.world_config['robots']]
    initial_poses = [[p, q] for p, q in zip(initial_base_positions, initial_base_quaternions)]
    initial_confs = [d['initial_joint_values'] for d in args.world_config['robots']]
    goal_confs = [d['goal_joint_values'] for d in args.world_config['robots']]
    urdf_path = 'assets/ur5/ur5.urdf'
    urdf_paths = [urdf_path] * len(args.world_config['robots'])

    # set up simulator
    pu.configure_pybullet(rendering=True, debug=True, yaw=58, pitch=-42, dist=1.4, target=(0, 0, 0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    obstacles = []
    for d in args.world_config['obstacles']:
        obstacles.append(p.loadURDF(fileName='assets/block.urdf',
                                    basePosition=d['initial_position'],
                                    baseOrientation=d['initial_quaternion'],
                                    useFixedBase=d['fixed_base']))
    obstacles += [plane]
    ur5_group = UR5Group(initial_poses, initial_confs, urdf_paths)

    # initial_pose = [[0, 0, 0.02], [0, 0, 0, 1]]
    # initial_joint_values = [-0.813358794499552, -0.37120422397572495, -0.754454729356351, 0, 0, 0]
    # ur5 = UR5(initial_pose, initial_joint_values, urdf_path)

    ## start and goal
    start_conf = ur5_group.get_joint_positions()
    start_poses = ur5_group.forward_kinematics(start_conf)
    goal_conf = list(itertools.chain.from_iterable(goal_confs))
    goal_poses = ur5_group.forward_kinematics(goal_conf)
    for pose in goal_poses:
        goal_marker = pu.draw_sphere_body(position=pose[0], radius=0.02, rgba_color=[1, 0, 0, 1])

    # place hoder to save the solution path
    path_conf = None

    # collision_fn = pu.get_collision_fn(ur5.id, UR5_JOINT_INDICES, obstacles=obstacles,
    #                                 attachments=[], self_collisions=True,
    #                                 disabled_collisions=set())

    collision_fn = ur5_group.get_collision_fn(obstacles=obstacles, attachments=[], self_collisions=True,
                                              disabled_collisions=set())
    extend_fn = ur5_group.get_extend_fn()

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
            path_conf = rrt.rrt(start=start_conf,
                                goal_sample=goal_conf,
                                distance=ur5_group.distance_fn,
                                sample=ur5_group.sample_fn,
                                extend=extend_fn,
                                collision=collision_fn,
                                goal_probability=0.2,
                                iterations=2000,
                                goal_test=goal_test,
                                visualize=True,
                                fk=ur5_group.forward_kinematics,
                                group=True)

            duration = time.time() - start_time
            print(duration)

    if path_conf is None:
        # no collision-free path is found within the time budget
        print("fail")
    else:
        # visualize the path and execute it
        for i in range(len(path_conf)):
            if i != len(path_conf) - 1:
                for pose1, pose2 in zip(ur5_group.forward_kinematics(path_conf[i]),
                                        ur5_group.forward_kinematics(path_conf[i + 1])):
                    pu.draw_line(pose1[0], pose2[0], rgb_color=[1, 0, 0], width=6)
        while True:
            for q in path_conf:
                ur5_group.set_joint_positions(q)
                time.sleep(0.5)
