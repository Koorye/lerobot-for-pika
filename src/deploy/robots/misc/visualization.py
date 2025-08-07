import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from abc import ABC, abstractmethod


class BaseTrajectoryRecorder(ABC):
    def __init__(self, init_state):
        self.init_state = init_state
        self.trajectory = {
            'timesteps': [],
            'positions': [],
            'euler_angles': [],
            'grips': [],
        }
        self.timestep = 0

    @abstractmethod
    def add(self, action):
        pass

    def clear(self):
        self.trajectory = {
            'timesteps': [],
            'positions': [],
            'euler_angles': [],
            'grips': [],
        }
        self.timestep = 0


class AbsoluteTrajectoryRecorder(BaseTrajectoryRecorder):
    def __init__(self, init_state):
        super().__init__(init_state)

    def add(self, action):
        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(action[:3])
        self.trajectory['euler_angles'].append(action[3:6])
        self.trajectory['grips'].append(action[6])
        self.timestep += 1


class DeltaGripperTrajectoryRecorder(BaseTrajectoryRecorder):
    def __init__(self, init_state):
        super().__init__(init_state)

        self.T_world_current = np.eye(4)
        self.T_world_current[:3, 3] = init_state[:3]
        rot = Rotation.from_euler('xyz', init_state[3:6])
        self.T_world_current[:3, :3] = rot.as_matrix()

    def add(self, action):
        dx, dy, dz = action[:3]
        drx, dry, drz = action[3:6]
        
        T_rel = np.eye(4)
        T_rel[:3, 3] = [dx, dy, dz]
        
        rel_rotation = Rotation.from_euler('xyz', [drx, dry, drz])
        rel_rot_matrix = rel_rotation.as_matrix()
        T_rel[:3, :3] = rel_rot_matrix @ T_rel[:3, :3]
        
        self.T_world_current = self.T_world_current @ T_rel
        
        global_position = self.T_world_current[:3, 3].copy()
        global_rot_matrix = self.T_world_current[:3, :3].copy()
        
        global_euler = Rotation.from_matrix(global_rot_matrix).as_euler('xyz')
        
        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(global_position.tolist())
        self.trajectory['euler_angles'].append(global_euler.tolist())
        self.trajectory['grips'].append(action[6])
        
        self.timestep += 1


class Visualizer:
    def __init__(self, image_names, traj_names, recoders, base_width=5, base_height=5):
        self.image_names = image_names
        self.num_images = len(image_names)
        self.traj_names = traj_names
        self.recorders = recoders
        self.base_width = base_width
        self.base_height = base_height

        self.images = None
        self.fig = None

    def add(self, images, actions):
        self.images = images
        for recorder, action in zip(self.recorders, actions):
            recorder.add(action)
    
    def create_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(self.base_width * (self.num_images + 1), self.base_height))

    def plot(self):
        if self.fig is None:
            self.create_plot()
        
        plt.clf()
        for i, image in enumerate(self.images):
            ax = self.fig.add_subplot(1, self.num_images + 1, i + 1)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(self.image_names[i])
        
        ax = self.fig.add_subplot(1, self.num_images + 1, self.num_images + 1, projection='3d')
        for name, recoder in zip(self.traj_names, self.recorders):
            positions = np.array(recoder.trajectory['positions'])
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    label=name, linewidth=2)
            
            ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                    'go', markersize=8)
            ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                    'ro', markersize=8)
            
            n_points = len(positions)
            step = max(1, n_points // 20) 
            euler_angles = np.array(recoder.trajectory['euler_angles'])

            for i in range(0, n_points, step):
                if i < n_points:
                    pos = positions[i]
                    direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                    ax.quiver(pos[0], pos[1], pos[2], 
                             direction[0], direction[1], direction[2],
                             length=0.1, color='r', alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.pause(0.001)


def get_visualizer(image_names, traj_names, init_states, traj_type='absolute'):
    if traj_type == 'absolute':
        recorders = [AbsoluteTrajectoryRecorder(init_state) for init_state in init_states]
    elif traj_type == 'delta_gripper':
        recorders = [DeltaGripperTrajectoryRecorder(init_state) for init_state in init_states]
    else:
        raise ValueError(f"Unsupported trajectory type: {traj_type}")

    visualizer = Visualizer(image_names, traj_names, recorders)
    return visualizer