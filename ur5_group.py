import numpy as np
from ur5 import UR5


class RobotGroup:
    def __init__(self, initial_poses, initial_confs, robot_urdfs):
        assert len(initial_poses) == len(robot_urdfs) == len(initial_confs)

        self.initial_poses = initial_poses
        self.initial_confs = initial_confs
        self.robot_urdfs = robot_urdfs
        self.num_robots = len(initial_poses)

        self.controllers = []
        self.robot_ids = []
        self.dof = 0
        for pose, conf, urdf in zip(self.initial_poses, self.initial_confs, self.robot_urdfs):
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
        q1_list = split(q1, self.num_robots)
        q2_list = split(q2, self.num_robots)
        for ctrl, q1_, q2_ in zip(self.controllers, q1_list, q2_list):
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

    def extend_fn(self, q1, q2):
        pass


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
