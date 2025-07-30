import math
import numpy as np


def deg_to_rad(deg):
    """Convert degrees to radians"""
    return deg * (math.pi / 180)


def rad_to_deg(rad):
    """Convert radians to degrees"""
    return rad * (180 / math.pi)


def deg001_to_deg(deg001):
    """Convert 0.001 degrees to degrees"""
    return deg001 / 1000.0


def deg_to_deg001(deg):
    """Convert degrees to 0.001 degrees"""
    return deg * 1000


def euler_to_rotation_matrix(rx, ry, rz):
    """Convert Euler angles (in radians) to rotation matrix (XYZ order)"""
    # XYZ order: first rx (roll), then ry (pitch), then rz (yaw)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    
    # Rotation matrix for XYZ order
    R = np.array([
        [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
        [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
        [-sy,   sx*cy,            cx*cy]
    ])
    return R


def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (XYZ order)"""
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    
    return np.array([rx, ry, rz])


def delta_to_absolute_gripper_translation(state_pos, state_rot, delta_pos, delta_rot):
    """
    Transform delta to absolute gripper translation,
    delta is in gripper frame and state is in world frame.
    
    Params:
    - delta: [x, y, z, rx, ry, rz] in gripper frame (x=forward, z=up)
             rx, ry, rz in 1 radian increments
    - state: [x, y, z, rx, ry, rz] in world frame (x=forward, z=up)
             rx, ry, rz in 1 radian increments
    Returns:
    - Absolute position and orientation in world frame [x, y, z, rx, ry, rz]
    """
    state_rot_matrix = euler_to_rotation_matrix(*state_rot)
    delta_rot_matrix = euler_to_rotation_matrix(*delta_rot)
    absolute_rot_matrix = state_rot_matrix @ delta_rot_matrix
    absolute_rot = rotation_matrix_to_euler(absolute_rot_matrix)

    absolute_pos = state_pos + state_rot_matrix.T @ delta_pos
    return absolute_pos, absolute_rot


def delta_to_absolute_gripper_translation_align_piper(state, delta):
    """
    For piper, the gripper should be rotated 90 degrees around the Y axis.
    """
    state = np.array(state)
    delta = np.array(delta)

    state_pos, state_rot = state[:3], state[3:6]
    state_rot = deg001_to_deg(state_rot)
    state_rot = deg_to_rad(state_rot)
    state_rot_matrix = euler_to_rotation_matrix(*state_rot)

    base_rot_delta = np.array([0, -np.pi/2, 0])  # 90 degrees around Y axis
    base_rot_delta_matrix = euler_to_rotation_matrix(*base_rot_delta)
    state_rot_matrix_aligned = state_rot_matrix @ base_rot_delta_matrix
    state_rot_aligned = rotation_matrix_to_euler(state_rot_matrix_aligned)

    delta_pos, delta_rot = delta[:3], delta[3:6]
    delta_rot = deg001_to_deg(delta_rot)
    delta_rot = deg_to_rad(delta_rot)

    new_state_pos, new_state_rot = delta_to_absolute_gripper_translation(state_pos, state_rot_aligned, delta_pos, delta_rot)
    new_state_rot_matrix = euler_to_rotation_matrix(*new_state_rot)
    base_rot_delta_inv = np.array([0, np.pi/2, 0])  # Inverse rotation
    base_rot_delta_inv_matrix = euler_to_rotation_matrix(*base_rot_delta_inv)
    new_state_rot_matrix_aligned = new_state_rot_matrix @ base_rot_delta_inv_matrix
    new_state_rot = rotation_matrix_to_euler(new_state_rot_matrix_aligned)

    new_state_rot = rad_to_deg(new_state_rot)
    new_state_rot = deg_to_deg001(new_state_rot)
    return np.concatenate([new_state_pos, new_state_rot, [delta[6]]]).astype(np.int64)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
                     #  x  y  z rx ry rz   grip
    state = np.array([  0, 0, 0, 0, 90000, 0, 60000])
    delta = np.array([100, 0, 0, 0, 10000, 0, 60000])
    print(delta_to_absolute_gripper_translation_align_piper(state, delta))

    # def degree_to_rad(degree):
    #     return degree * (math.pi / 180)
    
    # def rad_to_degree(rad):
    #     return rad * (180 / math.pi)

    # r1 = euler_to_rotation_matrix(0, degree_to_rad(10), 0)
    # r2 = euler_to_rotation_matrix(0, degree_to_rad(90), 0)
    # euler = rotation_matrix_to_euler(r1 @ r2)
    # euler = np.array([rad_to_degree(a) for a in euler])
    # print('euler:', euler)

    # states = []
    # for _ in range(100):
    #     state = delta_to_absolute_gripper_translation(state, delta)
    #     # state = delta_to_absolute_gripper_translation_align_piper(delta, state)
    #     states.append(state.copy())
    
    # # plot 3d x, y, z
    # states = np.array(states)[:, :3] / 1000  # Convert to mm
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(states[:, 0], states[:, 1], states[:, 2])
    # ax.set_xlabel('X (mm)')
    # ax.set_ylabel('Y (mm)')
    # ax.set_zlabel('Z (mm)')
    # ax.set_title('3D Trajectory of End Effector')

    # # x, y, z axis equal
    # # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for x, y, z axes
    # ax.axis('equal')
    # plt.show()
