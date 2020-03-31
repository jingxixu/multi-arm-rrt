import numpy as np
from ur5 import UR5
import pybullet_utils as pu
from rrt import rrt
from rrt_star import rrt_star, rrt_star_connect
from rrt_connect import birrt
from itertools import combinations, product
import time


class UR5Group:
    def __init__(self, initial_poses, initial_confs, urdf_paths):
        assert len(initial_poses) == len(urdf_paths) == len(initial_confs)

        self.initial_poses = initial_poses
        self.initial_confs = initial_confs
        self.urdf_paths = urdf_paths
        self.num_robots = len(initial_poses)

        self.controllers = []
        self.robot_ids = []
        self.dof = 0
        for pose, conf, urdf in zip(self.initial_poses, self.initial_confs, self.urdf_paths):
            controller = UR5(pose, conf, urdf)
            self.robot_ids.append(controller.id)
            self.controllers.append(controller)
            self.dof += len(controller.GROUP_INDEX['arm'])

    def set_joint_positions(self, joint_values):
        assert len(joint_values) == self.dof
        robot_joint_values = split(joint_values, self.num_robots)
        for c, jv in zip(self.controllers, robot_joint_values):
            c.set_arm_joints(jv)

    def get_joint_positions(self):
        joint_values = []
        for c in self.controllers:
            joint_values += c.get_arm_joint_values()
        return joint_values

    def difference_fn(self, q1, q2):
        difference = []
        split_q1 = split(q1, self.num_robots)
        split_q2 = split(q2, self.num_robots)
        for ctrl, q1_, q2_ in zip(self.controllers, split_q1, split_q2):
            difference += ctrl.arm_difference_fn(q1_, q2_)
        return difference

    def distance_fn(self, q1, q2):
        diff = np.array(self.difference_fn(q2, q1))
        return np.sqrt(np.dot(diff, diff))

    def sample_fn(self):
        values = []
        for ctrl in self.controllers:
            values += ctrl.arm_sample_fn()
        return values

    def get_extend_fn(self, resolutions=None):
        if resolutions is None:
            resolutions = 0.05 * np.ones(self.dof)

        def fn(q1, q2):
            steps = np.abs(np.divide(self.difference_fn(q2, q1), resolutions))
            num_steps = int(max(steps))
            waypoints = []
            diffs = self.difference_fn(q2, q1)
            for i in range(num_steps):
                waypoints.append(list(((float(i) + 1.0) / float(num_steps)) * np.array(diffs) + q1))
            return waypoints

        return fn

    def get_collision_fn(self, obstacles, attachments, self_collisions, disabled_collisions):
        # check_link_pairs is a 2d list
        check_link_pairs = []
        for i in range(self.num_robots):
            check_link_pairs.append(
                pu.get_self_link_pairs(self.robot_ids[i], self.controllers[i].GROUP_INDEX['arm'], disabled_collisions)
                if self_collisions else [])
        moving_bodies = self.robot_ids + [attachment.child for attachment in attachments]
        if obstacles is None:
            obstacles = list(set(pu.get_bodies()) - set(moving_bodies))
        check_body_pairs = list(product(moving_bodies, obstacles)) + list(combinations(moving_bodies, 2))

        def collision_fn(q):
            split_q = split(q, self.num_robots)
            for i, q_ in zip(range(self.num_robots), split_q):
                if pu.violates_limits(self.robot_ids[i], self.controllers[i].GROUP_INDEX['arm'], q_):
                    return True
            self.set_joint_positions(q)
            for attachment in attachments:
                attachment.assign()
            for i, pairs in enumerate(check_link_pairs):
                for link1, link2 in pairs:
                    if pu.pairwise_link_collision(self.robot_ids[i], link1, self.robot_ids[i], link2):
                        return True
            return any(pu.pairwise_collision(*pair) for pair in check_body_pairs)

        return collision_fn

    def forward_kinematics(self, q):
        """ return a list of eef poses """
        poses = []
        split_q = split(q, self.num_robots)
        for ctrl, q_ in zip(self.controllers, split_q):
            poses.append(ctrl.forward_kinematics(q_))
        return poses

    def plan_motion(self,
                    start_conf,
                    goal_conf,
                    planner='birrt',
                    smooth=200,
                    greedy=True,
                    goal_tolerance=0.001,
                    goal_bias=0.2,
                    resolutions=0.05,
                    iterations=2000,
                    restarts=10,
                    obstacles=[],
                    attachments=[],
                    self_collisions=True,
                    disabled_collisions=set()):

        # get some functions
        collision_fn = self.get_collision_fn(obstacles, attachments, self_collisions, disabled_collisions)
        goal_test = pu.get_goal_test_fn(goal_conf, goal_tolerance)
        extend_fn = self.get_extend_fn(resolutions)

        if planner == 'rrt':
            for i in range(restarts):
                iter_start = time.time()
                path_conf = rrt(start=start_conf,
                                goal_sample=goal_conf,
                                distance=self.distance_fn,
                                sample=self.sample_fn,
                                extend=extend_fn,
                                collision=collision_fn,
                                goal_probability=goal_bias,
                                iterations=iterations,
                                goal_test=goal_test,
                                greedy=greedy,
                                visualize=True,
                                fk=self.forward_kinematics,
                                group=True)
                iter_time = time.time() - iter_start
                if path_conf is None:
                    print('trial {} ({} iterations) fails in {:.2f} seconds'.format(i + 1, iterations, iter_time))
                    pu.remove_all_markers()
                else:
                    return path_conf
        elif planner == 'birrt':
            for i in range(restarts):
                iter_start = time.time()
                path_conf = birrt(start_conf=start_conf,
                                  goal_conf=goal_conf,
                                  distance=self.distance_fn,
                                  sample=self.sample_fn,
                                  extend=extend_fn,
                                  collision=collision_fn,
                                  iterations=iterations,
                                  smooth=smooth,
                                  visualize=True,
                                  fk=self.forward_kinematics,
                                  group=True,
                                  greedy=greedy)
                iter_time = time.time() - iter_start
                if path_conf is None:
                    print('trial {} ({} iterations) fails in {:.2f} seconds'.format(i + 1, iterations, iter_time))
                    pu.remove_all_markers()
                else:
                    return path_conf
        elif planner == 'rrt_star':
            for i in range(restarts):
                iter_start = time.time()
                path_conf = rrt_star(start=start_conf,
                                     goal=goal_conf,
                                     distance=self.distance_fn,
                                     sample=self.sample_fn,
                                     extend=extend_fn,
                                     collision=collision_fn,
                                     radius=0.15,
                                     goal_probability=goal_bias,
                                     informed=False,
                                     # iterations=iterations,
                                     # goal_test=goal_test,
                                     # greedy=greedy,
                                     visualize=True,
                                     fk=self.forward_kinematics,
                                     group=True
                                     )
                iter_time = time.time() - iter_start
                if path_conf is None:
                    print('trial {} ({} iterations) fails in {:.2f} seconds'.format(i + 1, iterations, iter_time))
                    pu.remove_all_markers()
                else:
                    return path_conf
        elif planner == 'birrt_star':
            for i in range(restarts):
                iter_start = time.time()
                path_conf = rrt_star_connect(start=start_conf,
                                             goal=goal_conf,
                                             distance=self.distance_fn,
                                             sample=self.sample_fn,
                                             extend=extend_fn,
                                             collision=collision_fn,
                                             radius=0.15,
                                             visualize=True,
                                             fk=self.forward_kinematics,
                                             group=True,
                                             max_time=40)
                iter_time = time.time() - iter_start
                if path_conf is None:
                    print('trial {} ({} iterations) fails in {:.2f} seconds'.format(i + 1, iterations, iter_time))
                    pu.remove_all_markers()
                else:
                    return path_conf
        else:
            raise ValueError('planner must be in \'rrt\' or \'birrt\'')


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
