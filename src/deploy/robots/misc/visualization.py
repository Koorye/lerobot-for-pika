import copy
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation


class BaseTrajectoryVisualizer(ABC):
    def __init__(self, state):
        super().__init__()

        self.init_position = state[:3]
        self.init_euler = state[3:6]
        self.init_grip = state[6]

        self.trajectory = {
            'timesteps': [],
            'positions': [],
            'euler_angles': [],
            'grips': [],
        }
        self.timestep = 0

        plt.ion()  # Enable interactive mode for live updates
        self.fig = plt.figure(figsize=(10, 10))

    @abstractmethod
    def add(self, state):
        pass

    def reset(self, state):
        position, euler, grip = state[:3], state[3:6], state[6]
        self.__init__(
            init_position=position, 
            init_euler=euler, 
            init_grip=grip
        )

    def plot(self):
        plt.clf()
        positions = np.array(self.trajectory['positions'])
        
        ax = self.fig.add_subplot(111, projection='3d')
        
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2)
        
        ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                'go', markersize=8, label='start')
        ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                'ro', markersize=8, label='end')
        
        n_points = len(positions)
        step = max(1, n_points // 20) 
        
        euler_angles = np.array(self.trajectory['euler_angles'])
        
        for i in range(0, n_points, step):
            if i < n_points:
                pos = positions[i]
                direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                ax.quiver(pos[0], pos[1], pos[2], 
                         direction[0], direction[1], direction[2],
                         length=0.1, color='r', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.pause(1e-4)


class MultiTrajectoriesVisualizer:
    def __init__(
            self, 
            names,
            visualizers,
        ):
        self.names = names
        self.visualizers = visualizers
        self.fig = plt.figure(figsize=(10, 10))

    def add(self, states):
        for name, state, converter in zip(self.names, states, self.visualizers):
            converter.add(state)
    
    def reset(self, states):
        for name, state, converter in zip(self.names, states, self.visualizers):
            converter.reset(state)

    def plot_trajectories_3d(self, names=None):
        plt.clf()
        ax = self.fig.add_subplot(111, projection='3d')
        
        for name, converter in zip(self.names, self.converters):
            if names is not None:
                if name not in names:
                    continue
            
            positions = np.array(converter.trajectory['positions'])
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    label=name, linewidth=2)
            
            ax.plot(positions[0, 0], positions[0, 1], positions[0, 2], 
                    'go', markersize=8)
            ax.plot(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                    'ro', markersize=8)
            
            n_points = len(positions)
            step = max(1, n_points // 20) 
            euler_angles = np.array(converter.trajectory['euler_angles'])

            for i in range(0, n_points, step):
                if i < n_points:
                    pos = positions[i]
                    direction = Rotation.from_euler('xyz', euler_angles[i]).apply([1, 0, 0])
                    ax.quiver(pos[0], pos[1], pos[2], 
                             direction[0], direction[1], direction[2],
                             length=0.1, color='r', alpha=0.7)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.tight_layout()
        plt.axis('equal')
        plt.pause(1e-4)


class AbsoluteTrajectoryVisualizer(BaseTrajectoryVisualizer):
    def __init__(self, ee_state):
        super().__init__(ee_state)

    def add(self, state):
        position, euler, grip = state[:3], state[3:6], state[6]
        self.trajectory['timesteps'].append(self.timestep)
        self.trajectory['positions'].append(position)
        self.trajectory['euler_angles'].append(euler)
        self.trajectory['grips'].append(grip)
        self.timestep += 1


class DeltaGripperTrajectoryVisualizer(BaseTrajectoryVisualizer):
    def __init__(self, ee_state):
        super().__init__(ee_state)

        self.T_world_current = np.eye(4)
        self.T_world_current[:3, 3] = self.init_position
        rot = Rotation.from_euler('xyz', self.init_euler)
        self.T_world_current[:3, :3] = rot.as_matrix()
    
    def add(self, state):
        position, euler, grip = state[:3], state[3:6], state[6]
        dx, dy, dz = position
        drx, dry, drz = euler
        
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
        self.trajectory['grips'].append(grip)
        self.timestep += 1


def get_visualizer(ee_state, control_mode, multi_arm=False) -> BaseTrajectoryVisualizer:
    if control_mode == 'ee_absolute':
        visualizer = AbsoluteTrajectoryVisualizer(ee_state)
    elif control_mode == 'ee_delta_gripper':
        visualizer = DeltaGripperTrajectoryVisualizer(ee_state)
    else:
        raise ValueError(f"Unsupported control mode: {control_mode}")
    
    if multi_arm:
        return MultiTrajectoriesVisualizer([
            'arm_left', 
            'arm_right'
        ], [
            copy.deepcopy(visualizer), 
            copy.deepcopy(visualizer)
        ])
    else:
        return visualizer
