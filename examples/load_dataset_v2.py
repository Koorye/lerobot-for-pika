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


class PikaTrajectoryConverter:
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

        self.trajectory = {
            'timesteps': [],
            'positions': [],
            'euler_angles': [],
            'grips': [],
        }
        self.timestep = 0

    def add_frame(self, position, euler, grip):
        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(position)
        self.trajectory['euler_angles'].append(euler)
        self.trajectory['grips'].append(grip)
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
        self.names = names
        self.converters = []
        for name, pos, euler, grip in zip(names, init_positions, init_eulers, init_grips):
            converter = PikaTrajectoryConverter(pos, euler, grip)
            self.converters.append(converter)

    def add_frame(self, positions, eulers, grips):
        for converter, pos, euler, grip in zip(self.converters, positions, eulers, grips):
            converter.add_frame(pos, euler, grip)
    
    def get_trajectories(self):
        trajectories = {}
        for name, converter in zip(self.names, self.converters):
            trajectories[name] = converter.get_trajectory()
        return trajectories
    
    def reset(self, init_positions, init_eulers, init_grips):
        for converter, pos, euler, grip in zip(self.converters, init_positions, init_eulers, init_grips):
            converter.reset(pos, euler, grip)

    def plot_trajectories_3d(self, figsize=(10, 8), names=None):
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

    for sample in tqdm(dataset, desc="Processing samples"):
        episode = sample['episode_index']
        states = sample['states'].numpy()
        left_position, left_euler, left_grip = states[:3], states[3:6], states[6]
        right_position, right_euler, right_grip = states[7:10], states[10:13], states[13]

        if episode != current_episode:
            current_episode = episode

            if converter is not None:
                converter.plot_trajectories_3d(names=['left_gripper', 'right_gripper'])
            
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
            converter.add_frame(
                positions=positions, 
                eulers=eulers, 
                grips=grips,
            )
    
    if converter is not None:
        converter.plot_trajectories_3d(names=['left_gripper', 'right_gripper'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pika trajectory data to absolute world coordinates.")
    parser.add_argument('--repo-id', type=str, required=True, help='The repository ID of the dataset.')
    args = parser.parse_args()
    main(args)
