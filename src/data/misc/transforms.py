"""
This module is used to convert absolute position coordinates in the world coordinate system 
into various relative representations, including:
1. Delta base coordinates (action is delta position and orientation relative to the robot base, x always points forward)
2. Delta gripper coordinates (action is delta position and orientation relative to the robot gripper's local frame, x points gripper's forward direction)
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(roll, pitch, yaw):
    return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()


def rotation_matrix_to_euler(matrix):
    return Rotation.from_matrix(matrix).as_euler('xyz')


class BaseTransform(ABC):
    """
    Base class for end effector transforms.
    """

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __call__(self, end_effector_state, next_end_effector_state):
        """
        Transform the end effector state and action to a standardized format.
        """
        pass


class AbsoluteTransform(BaseTransform):
    """
    Transform that converts end effector actions to absolute world coordinates (no change).
    """

    def __call__(self, end_effector_state, next_end_effector_state):
        return next_end_effector_state


class AbsoluteToDeltaBaseTransform(BaseTransform):
    """
    Transform that converts absolute end effector state to delta base coordinates.
    """

    def __call__(self, end_effector_state, next_end_effector_state):
        current_pos, current_euler = end_effector_state[:3], end_effector_state[3:6]
        next_pos, next_euler , gripper = next_end_effector_state[:3], next_end_effector_state[3:6], next_end_effector_state[6]
        delta_pos = next_pos - current_pos
        delta_euler = next_euler - current_euler
        return np.concatenate((delta_pos, delta_euler, np.array([gripper])), axis=0)


class AbsoluteToDeltaGripperTransform(BaseTransform):
    """
    Transform that converts absolute end effector state to delta gripper coordinates.
    """
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, end_effector_state, next_end_effector_state):
        current_pos, current_euler = end_effector_state[:3], end_effector_state[3:6]
        next_pos, next_euler, gripper = next_end_effector_state[:3], next_end_effector_state[3:6], next_end_effector_state[6]

        current_rot_matrix = euler_to_rotation_matrix(*current_euler)
        next_rot_matrix = euler_to_rotation_matrix(*next_euler)
        delta_rot_matrix = next_rot_matrix @ current_rot_matrix.T
        delta_euler = rotation_matrix_to_euler(delta_rot_matrix)

        delta_pos = current_rot_matrix.T @ (next_pos - current_pos)

        return np.concatenate((delta_pos, delta_euler, np.array([gripper])), axis=0)


class BiTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, end_effector_state, next_end_effector_state):
        left_end_effector_state = end_effector_state[:7]
        right_end_effector_state = end_effector_state[7:]
        left_next_end_effector_state = next_end_effector_state[:7]
        right_next_end_effector_state = next_end_effector_state[7:]
        left_transformed = self.transform(left_end_effector_state, left_next_end_effector_state)
        right_transformed = self.transform(right_end_effector_state, right_next_end_effector_state)
        return np.concatenate((left_transformed, right_transformed), axis=0)


def get_transform(transform_type, multi_arm=True):
    if transform_type == "ee_absolute":
        transform =  AbsoluteTransform()
    elif transform_type == "ee_delta_base":
        transform = AbsoluteToDeltaBaseTransform()
    elif transform_type == "ee_delta_gripper":
        transform = AbsoluteToDeltaGripperTransform()
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    if multi_arm:
        transform = BiTransform(transform)
    
    return transform