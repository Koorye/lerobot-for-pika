import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation


def euler_to_rotation_matrix(roll, pitch, yaw):
    return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()


def rotation_matrix_to_euler(matrix):
    return Rotation.from_matrix(matrix).as_euler('xyz')


class BaseTransform(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __call__(self, end_effector_state, end_effector_action):
        """
        Transform the end effector state and action to a standardized format.
        """
        pass


class AbsoluteTransform(BaseTransform):
    def __call__(self, end_effector_state, end_effector_action):
        return end_effector_action


class DeltaBaseToAbsoluteTransform(BaseTransform):
    def __call__(self, end_effector_state, end_effector_action):
        current_pos, current_euler = end_effector_state[:3], end_effector_state[3:6]
        delta_pos, delta_euler, gripper = end_effector_action[:3], end_effector_action[3:6], end_effector_action[6]

        absolute_pos = current_pos + delta_pos
        absolute_euler = current_euler + delta_euler

        return list(np.concatenate((absolute_pos, absolute_euler, np.array([gripper])), axis=0))


class DeltaGripperToAbsoluteTransform(BaseTransform):
    def __init__(self, base_euler=None):
        super().__init__()
        self.base_euler = base_euler
    
    def __call__(self, end_effector_state, end_effector_action):
        current_pos, current_euler = end_effector_state[:3], end_effector_state[3:6]
        delta_pos, delta_euler, gripper = end_effector_action[:3], end_effector_action[3:6], end_effector_action[6]

        current_rot_matrix = euler_to_rotation_matrix(*current_euler)

        if self.base_euler is not None:
            align_rot_matrix = euler_to_rotation_matrix(*self.base_euler)
            current_rot_matrix = current_rot_matrix @ align_rot_matrix.T

        delta_rot_matrix = euler_to_rotation_matrix(*delta_euler)
        absolute_rot_matrix = current_rot_matrix @ delta_rot_matrix

        if self.base_euler is not None:
            absolute_rot_matrix = absolute_rot_matrix @ align_rot_matrix
        
        absolute_euler = rotation_matrix_to_euler(absolute_rot_matrix)

        absolute_pos = current_pos + current_rot_matrix.T @ delta_pos

        return list(np.concatenate((absolute_pos, absolute_euler, np.array([gripper])), axis=0))


def get_transform(transform_type, base_euler=None):
    if transform_type == "ee_absolute":
        return AbsoluteTransform()
    elif transform_type == "ee_delta_base_to_absolute":
        return DeltaBaseToAbsoluteTransform()
    elif transform_type == "ee_delta_gripper_to_absolute":
        return DeltaGripperToAbsoluteTransform(base_euler=base_euler)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")