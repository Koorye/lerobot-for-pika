import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.1'
except:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.0'


def compute_delta_euler(euler_prev, euler_curr, rotation_order='xyz'):
    """
    计算两帧之间的相对旋转变化
    
    参数:
        euler_prev: 前一帧欧拉角 (rx, ry, rz) 单位:弧度
        euler_curr: 当前帧欧拉角 (rx, ry, rz) 单位:弧度
        rotation_order: 旋转顺序 ('xyz', 'zyx'等)
    
    返回:
        relative_euler: 相对旋转变化 (drx, dry, drz) 单位:弧度
    """
    # 将欧拉角转换为旋转对象
    rot_prev = Rotation.from_euler(rotation_order, euler_prev)
    rot_curr = Rotation.from_euler(rotation_order, euler_curr)
    
    # 获取旋转矩阵
    R_prev = rot_prev.as_matrix()
    R_curr = rot_curr.as_matrix()
    
    # 计算相对旋转矩阵: R_rel = R_prev⁻¹ × R_curr
    R_rel = np.linalg.inv(R_prev) @ R_curr
    
    # 将相对旋转矩阵转换为欧拉角
    relative_rot = Rotation.from_matrix(R_rel)
    relative_euler = relative_rot.as_euler(rotation_order)
    
    return relative_euler


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


class PikaTrajectoryConverter:
    """
    将手持式夹爪采集的相对运动数据转换为固定世界坐标系中的绝对轨迹
    
    功能：
    1. 处理带有相对位置和角度变化的帧数据
    2. 考虑X轴始终指向夹爪朝向的动态特性
    3. 将数据转换为固定世界坐标系
    4. 提供轨迹可视化和数据导出功能
    
    使用方法：
    converter = HandheldGripperTrajectoryConverter(init_position, init_euler)
    for frame_data in collected_data:
        global_pose = converter.process_frame(frame_data)
        trajectory = converter.get_trajectory()
    
    """
    
    def __init__(
            self, 
            init_position, 
            init_euler, 
            init_grip
        ):
        """
        初始化转换器
        :param init_position: 初始位置 (x, y, z) 单位:米
        :param init_euler: 初始欧拉角 (rx, ry, rz) 单位:弧度
        """
        self.init_position = init_position
        self.init_euler = init_euler
        self.init_grip = init_grip

        # 初始化变换矩阵（世界坐标系到夹爪初始位姿）
        self.T_world_current = np.eye(4)
        
        self.T_world_current[:3, 3] = init_position
            
        rot = Rotation.from_euler('xyz', init_euler)
        self.T_world_current[:3, :3] = rot.as_matrix()

        self.base_pose = np.concatenate([init_position, init_euler]).copy()
        self.end_pose = np.concatenate([init_position, init_euler]).copy()
            
        # 存储轨迹历史
        self.trajectory = {
            'timesteps': [],
            'positions': [],
            'rotations': [],
            'euler_angles': [],
            'grips': [],
        }
        
        # 记录处理帧数
        self.timestep = 0
    
    def add_frame(self, position, euler, grip):
        begin_matrix = create_transformation_matrix(*self.base_pose)
        zero_matrix = create_transformation_matrix(*self.end_pose)
        end_matrix = create_transformation_matrix(*np.concatenate([position, euler]))
        result_matrix = np.dot(zero_matrix, np.dot(np.linalg.inv(begin_matrix), end_matrix))
        x, y, z, rx, ry, rz = matrix_to_xyzrpy(result_matrix)
        self.end_pose = np.array([x, y, z, rx, ry, rz]).copy()

        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append([x, y, z])
        self.trajectory['euler_angles'].append([rx, ry, rz])
        self.timestep += 1
        
    def add_delta_frame(self, delta_position, delta_euler, grip):
        """
        处理一帧数据并返回当前全局位姿
        :param frame_data: 相对变换数据 [dx, dy, dz, drx, dry, drz, grip]
                           dx,dy,dz: 相对位移 (米)
                           drx,dry,drz: 相对欧拉角变化 (弧度)
                           grip: 夹爪握持状态
        :return: 当前全局位姿 (位置, 旋转矩阵, 欧拉角)
        """
        # 解析帧数据
        dx, dy, dz = delta_position
        drx, dry, drz = delta_euler
        
        # 构建相对齐次变换矩阵 (相对于当前夹爪坐标系)
        T_rel = np.eye(4)
        
        # 添加平移分量
        T_rel[:3, 3] = [dx, dy, dz]
        
        # 创建旋转对象 (绕当前坐标系轴的旋转)
        rel_rotation = Rotation.from_euler('xyz', [drx, dry, drz])
        rel_rot_matrix = rel_rotation.as_matrix()
        
        # 应用旋转 (注意：X轴始终指向夹爪朝向)
        # 将旋转应用到变换矩阵的正确位置
        T_rel[:3, :3] = rel_rot_matrix @ T_rel[:3, :3]
        
        # 更新全局变换矩阵 (注意乘法的顺序)
        self.T_world_current = self.T_world_current @ T_rel
        self.T_world_current[:3, :3] = rel_rot_matrix
        
        # 提取当前全局位姿
        global_position = self.T_world_current[:3, 3].copy()
        global_rot_matrix = self.T_world_current[:3, :3].copy()
        
        # 转换为欧拉角 (xyz顺序)
        global_euler = Rotation.from_matrix(global_rot_matrix).as_euler('xyz')
        
        # 存储轨迹
        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(global_position.tolist())
        self.trajectory['rotations'].append(global_rot_matrix.tolist())
        self.trajectory['euler_angles'].append(global_euler.tolist())
        
        self.timestep += 1
        
    def get_trajectory(self):
        """获取完整的转换后轨迹"""
        return self.trajectory
    
    def reset(self, init_position, init_euler, init_grip):
        self.__init__(
            init_position=init_position, 
            init_euler=init_euler, 
            init_grip=init_grip
        )
    
    def plot_trajectory_3d(self, figsize=(10, 8)):
        """绘制3D轨迹图"""
        positions = np.array(self.trajectory['positions'])
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹线
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2)
        
        # 在起点和终点添加特殊标记
        ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                'go', markersize=8, label='start')
        ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                'ro', markersize=8, label='end')
        
        # 每隔N个点绘制方向箭头
        n_points = len(positions)
        step = max(1, n_points // 20)  # 绘制20个箭头或更少
        
        euler_angles = np.array(self.trajectory['euler_angles'])
        
        for i in range(0, n_points, step):
            if i < n_points:
                pos = positions[i]
                # 计算方向向量 (X轴方向)
                direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                
                # 绘制箭头
                ax.quiver(pos[0], pos[1], pos[2], 
                         direction[0], direction[1], direction[2],
                         length=0.1, color='r', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.show()
    
    def export_trajectory_csv(self, filename):
        """导出轨迹数据到CSV文件"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写标题行
            writer.writerow(['timestamp', 'pos_x', 'pos_y', 'pos_z', 
                            'euler_rx', 'euler_ry', 'euler_rz', 'grip'])
            
            # 写数据行
            for i in range(len(self.trajectory['frame_ids'])):
                timestamp = self.trajectory['timestamps'][i]
                pos = self.trajectory['positions'][i]
                euler = self.trajectory['euler_angles'][i]
                grip = self.trajectory['grips'][i]
                
                writer.writerow([
                    timestamp,
                    pos[0], pos[1], pos[2],
                    euler[0], euler[1], euler[2],
                    grip,
                ])


class MultiPikaTrajectoryConverter:
    """
    多夹爪轨迹转换器，处理多个夹爪的相对运动数据转换为固定世界坐标系中的绝对轨迹
    """
    
    def __init__(
            self, 
            init_positions, 
            init_eulers, 
            init_grips, 
            names,
        ):
        """
        初始化转换器
        :param init_positions: 初始位置列表 [[x1, y1, z1], [x2, y2, z2], ...] 单位:米
        :param init_eulers: 初始欧拉角列表 [[rx1, ry1, rz1], [rx2, ry2, rz2], ...] 单位:弧度
        :param init_grips: 初始夹爪握持状态列表 [grip1, grip2, ...]
        """
        self.init_positions = init_positions
        self.init_eulers = init_eulers
        self.init_grips = init_grips
        self.names = names
        
        self.converters = []
        
        for name, pos, euler, grip in zip(names, init_positions, init_eulers, init_grips):
            converter = PikaTrajectoryConverter(pos, euler, grip)
            self.converters.append(converter)

    def add_frame(self, positions, eulers, grips):
        """
        处理一帧数据并返回当前全局位姿
        :param delta_positions: 相对变换数据列表 [[dx1, dy1, dz1], [dx2, dy2, dz2], ...] 单位:米
        :param delta_eulers: 相对欧拉角变化列表 [[drx1, dry1, drz1], [drx2, dry2, drz2], ...] 单位:弧度
        :param grips: 夹爪握持状态列表 [grip1, grip2, ...]
        :return: 当前全局位姿列表 [(position1, rotation_matrix1), (position2, rotation_matrix2), ...]
        """
        global_poses = []
        
        for converter, pos, euler, grip in zip(self.converters, positions, eulers, grips):
            global_pose = converter.add_frame(pos, euler, grip)
            global_poses.append(global_pose)
        
        return global_poses
            
    def add_delta_frame(self, delta_positions, delta_eulers, grips):
        """
        处理一帧数据并返回当前全局位姿
        :param delta_positions: 相对变换数据列表 [[dx1, dy1, dz1], [dx2, dy2, dz2], ...] 单位:米
        :param delta_eulers: 相对欧拉角变化列表 [[drx1, dry1, drz1], [drx2, dry2, drz2], ...] 单位:弧度
        :param grips: 夹爪握持状态列表 [grip1, grip2, ...]
        :return: 当前全局位姿列表 [(position1, rotation_matrix1), (position2, rotation_matrix2), ...]
        """
        global_poses = []
        
        for converter, delta_pos, delta_euler, grip in zip(self.converters, delta_positions, delta_eulers, grips):
            global_pose = converter.add_delta_frame(delta_pos, delta_euler, grip)
            global_poses.append(global_pose)
        
        return global_poses
    
    def get_trajectories(self):
        """获取所有夹爪的完整轨迹"""
        trajectories = {}
        
        for name, converter in zip(self.names, self.converters):
            trajectories[name] = converter.get_trajectory()
        
        return trajectories
    
    def reset(self, init_positions, init_eulers, init_grips):
        """
        重置所有夹爪的初始状态
        :param init_positions: 初始位置列表 [[x1, y1, z1], [x2, y2, z2], ...] 单位:米
        :param init_eulers: 初始欧拉角列表 [[rx1, ry1, rz1], [rx2, ry2, rz2], ...] 单位:弧度
        :param init_grips: 初始夹爪握持状态列表 [grip1, grip2, ...]
        """
        self.__init__(init_positions, init_eulers, init_grips, self.names)

    def plot_trajectories_3d(self, figsize=(10, 8), names=None):
        """绘制所有夹爪的3D轨迹图"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        for name, converter in zip(self.names, self.converters):
            if names is not None:
                if name not in names:
                    continue
            
            positions = np.array(converter.trajectory['positions'])
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    label=name, linewidth=2)
            
            # 在起点和终点添加特殊标记
            ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                    'go', markersize=8)
            ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                    'ro', markersize=8)
            
            n_points = len(positions)
            step = max(1, n_points // 20)  # 绘制20
            euler_angles = np.array(converter.trajectory['euler_angles'])

            for i in range(0, n_points, step):
                if i < n_points:
                    pos = positions[i]
                    # 计算方向向量 (X轴方向)
                    direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                    
                    # 绘制箭头
                    ax.quiver(pos[0], pos[1], pos[2], 
                             direction[0], direction[1], direction[2],
                             length=0.1, color='r', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.show()


def main(args):
    dataset = LeRobotDataset(args.repo_id)
    current_episode = None
    converter = None
    past_positions = None
    past_eulers = None

    for sample in tqdm(dataset, desc="Processing samples"):
        episode = sample['episode_index']
        states = sample['states'].numpy()
        left_position, left_euler, left_grip = states[:3], states[3:6], states[6]
        right_position, right_euler, right_grip = states[7:10], states[10:13], states[13]

        if episode != current_episode:
            current_episode = episode
            past_positions = [left_position, right_position]
            past_eulers = [left_euler, right_euler]

            converter = MultiPikaTrajectoryConverter(
                init_positions=[left_position, right_position],
                init_eulers=[left_euler, right_euler],
                init_grips=[left_grip, right_grip],
                names=['left_gripper', 'right_gripper'],
            )
        
        else:
            positions = [left_position, right_position]
            eulers = [left_euler, right_euler]
            grips = [left_grip, right_grip]

            delta_positions = [
                left_position - past_positions[0],
                right_position - past_positions[1]
            ]
            delta_eulers = [
                compute_delta_euler(past_eulers[0], left_euler),
                compute_delta_euler(past_eulers[1], right_euler),
            ]
            past_positions = [left_position, right_position]
            past_eulers = [left_euler, right_euler]
            delta_eulers = [np.array([0, 0, 0]), np.array([0, 0, 0])]

            # converter.add_frame(
            #     positions=delta_positions, 
            #     eulers=delta_eulers, 
            #     grips=grips,
            # )

            converter.add_delta_frame(
                delta_positions=delta_positions, 
                delta_eulers=delta_eulers, 
                grips=grips,
            )

    converter.plot_trajectories_3d(names=['left_gripper', 'right_gripper'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pika trajectory data to absolute world coordinates.")
    parser.add_argument('--repo-id', type=str, required=True, help='The repository ID of the dataset.')
    args = parser.parse_args()
    main(args)
