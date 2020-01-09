import os
import pybullet as p
import numpy as np
import pybullet_utils as pu
from math import pi


class UR5:
    GROUPS = {
        'arm': ["shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"],
        'gripper': None
    }

    GROUP_INDEX = {
        'arm': [1, 2, 3, 4, 5, 6],
        'gripper': None
    }

    INDEX_NAME_MAP = {
        0: 'world_joint',
        1: 'shoulder_pan_joint',
        2: 'shoulder_lift_joint',
        3: 'elbow_joint',
        4: 'wrist_1_joint',
        5: 'wrist_2_joint',
        6: 'wrist_3_joint',
        7: 'ee_fixed_joint',
        8: 'wrist_3_link-tool0_fixed_joint',
        9: 'base_link-base_fixed_joint'
    }

    LOWER_LIMITS = [-2 * pi, -2 * pi, -pi, -2 * pi, -2 * pi, -2 * pi]
    UPPER_LIMITS = [2 * pi, 2 * pi, pi, 2 * pi, 2 * pi, 2 * pi]
    MAX_VELOCITY = [3.15, 3.15, 3.15, 3.2, 3.2, 3.2]
    MAX_FORCE = [150.0, 150.0, 150.0, 28.0, 28.0, 28.0]

    HOME = [0, 0, 0, 0, 0, 0]
    UP = [0, -1.5707, 0, -1.5707, 0, 0]
    RESET = [0, -1, 1, 0.5, 1, 0]
    EEF_LINK_INDEX = 7

    def __init__(self,
                 initial_pose,
                 initial_joint_values,
                 urdf_path,
                 speed,
                 time_step):
        self.initial_pose = initial_pose
        self.initial_joint_values = initial_joint_values
        self.urdf_path = urdf_path
        self.speed = speed
        self.time_step = time_step

        self.id = p.loadURDF(self.urdf_path,
                             basePosition=self.initial_pose[0],
                             baseOrientation=self.initial_pose[1],
                             flags=p.URDF_USE_SELF_COLLISION)
        self.set_arm_joints(self.initial_joint_values)

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['arm'], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def get_arm_joint_values(self):
        return pu.get_joint_positions(self.id, self.GROUP_INDEX['arm'])

    def get_eef_pose(self, link=EEF_LINK_INDEX):
        return pu.get_link_pose(self.id, link)

    def plan_arm_simple(self, start_joint_values, goal_joint_values):
        start_joint_values = np.array(start_joint_values)
        goal_joint_values = np.array(goal_joint_values)

        num_steps = int(np.max(goal_joint_values - start_joint_values) / (self.speed * self.time_step))
        waypoints = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return waypoints

    def inverse_kinematics(self, position, orientation=None):
        return pu.inverse_kinematics(self.id, self.EEF_LINK_INDEX, position, orientation)

    def forward_kinematics(self, joint_values):
        return pu.forward_kinematics(self.id, self.GROUP_INDEX['arm'], joint_values, self.EEF_LINK_INDEX)

    def sample_joint_values(self):
        return np.random.uniform(low=self.LOWER_LIMITS, high=self.UPPER_LIMITS)