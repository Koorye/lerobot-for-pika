import numpy as np
from scipy.spatial.transform import Rotation


def compute_relative_pose(prev_pos, prev_euler, curr_pos, curr_euler, rotation_order='xyz'):
    """
    计算当前帧相对于上一帧的相对位置和旋转变化
    
    参数:
        prev_state: 上一帧状态 [x, y, z, rx, ry, rz]
        current_state: 当前帧状态 [x, y, z, rx, ry, rz]
        rotation_order: 欧拉角的旋转顺序 ('xyz', 'zyx'等)
    
    返回:
        relative_pose: 相对位姿变化 [dx, dy, dz, drx, dry, drz]
    """
    # 1. 构建上一帧的齐次变换矩阵
    T_prev = np.eye(4)
    T_prev[:3, 3] = prev_pos
    T_prev[:3, :3] = Rotation.from_euler(rotation_order, prev_euler).as_matrix()
    
    # 2. 构建当前帧的齐次变换矩阵
    T_curr = np.eye(4)
    T_curr[:3, 3] = curr_pos
    T_curr[:3, :3] = Rotation.from_euler(rotation_order, curr_euler).as_matrix()
    
    # 3. 计算相对变换矩阵 (在上一帧坐标系中)
    T_relative = np.linalg.inv(T_prev) @ T_curr
    
    # 4. 提取相对位置变化
    dx, dy, dz = T_relative[:3, 3]
    
    # 5. 提取相对旋转变化
    relative_rot = Rotation.from_matrix(T_relative[:3, :3])
    drx, dry, drz = relative_rot.as_euler(rotation_order)
    
    return np.array([dx, dy, dz]), np.array([drx, dry, drz])

# 增强版：处理角度跳变问题
def robust_compute_relative_pose(prev_pos, prev_euler, curr_pos, curr_euler, rotation_order='xyz'):
    """
    鲁棒的相对位姿计算，处理角度跳变问题
    
    参数:
        prev_state: 上一帧状态 [x, y, z, rx, ry, rz]
        current_state: 当前帧状态 [x, y, z, rx, ry, rz]
        rotation_order: 欧拉角的旋转顺序 ('xyz', 'zyx'等)
    
    返回:
        relative_pose: 相对位姿变化 [dx, dy, dz, drx, dry, drz]
    """
    # 1. 构建上一帧的齐次变换矩阵
    rot_prev = Rotation.from_euler(rotation_order, prev_euler)
    T_prev = np.eye(4)
    T_prev[:3, 3] = prev_pos
    T_prev[:3, :3] = rot_prev.as_matrix()
    
    # 2. 构建当前帧的齐次变换矩阵
    rot_curr = Rotation.from_euler(rotation_order, curr_euler)
    T_curr = np.eye(4)
    T_curr[:3, 3] = curr_pos
    T_curr[:3, :3] = rot_curr.as_matrix()
    
    # 3. 计算相对变换矩阵 (在上一帧坐标系中)
    T_relative = np.linalg.inv(T_prev) @ T_curr
    
    # 4. 提取相对位置变化
    dx, dy, dz = T_relative[:3, 3]
    
    # 5. 使用四元数计算相对旋转变化（避免角度跳变）
    quat_prev = rot_prev.as_quat()
    quat_curr = rot_curr.as_quat()
    
    # 计算相对四元数: q_rel = q_prev⁻¹ ⊗ q_curr
    q_prev_inv = np.array([quat_prev[3], -quat_prev[0], -quat_prev[1], -quat_prev[2]])  # [w, -x, -y, -z]
    q_rel = np.array([
        q_prev_inv[0]*quat_curr[3] - q_prev_inv[1]*quat_curr[0] - q_prev_inv[2]*quat_curr[1] - q_prev_inv[3]*quat_curr[2],
        q_prev_inv[0]*quat_curr[0] + q_prev_inv[1]*quat_curr[3] + q_prev_inv[2]*quat_curr[2] - q_prev_inv[3]*quat_curr[1],
        q_prev_inv[0]*quat_curr[1] - q_prev_inv[1]*quat_curr[2] + q_prev_inv[2]*quat_curr[3] + q_prev_inv[3]*quat_curr[0],
        q_prev_inv[0]*quat_curr[2] + q_prev_inv[1]*quat_curr[1] - q_prev_inv[2]*quat_curr[0] + q_prev_inv[3]*quat_curr[3]
    ])
    
    # 将相对四元数转换为欧拉角
    relative_rot = Rotation.from_quat([q_rel[1], q_rel[2], q_rel[3], q_rel[0]])  # [x, y, z, w]
    drx, dry, drz = relative_rot.as_euler(rotation_order)
    
    return np.array([dx, dy, dz]), np.array([drx, dry, drz])
