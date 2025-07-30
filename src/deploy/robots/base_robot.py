import matplotlib.pyplot as plt
import numpy as np

from ...utils.standardlizations import get_standardization

_DEFAULT_INIT_STATES = [0, 0, 0, 0, 0, 0, 0] # x, y, z, rx, ry, rz, grip


class BaseRobot(object):
    def __init__(
        self, 
        name='base',
        control_mode='eef_absolute',
        use_standardization=True,
        init_states=_DEFAULT_INIT_STATES,
    ):
        self.name = name
        self.control_mode = control_mode
        self.use_standardization = use_standardization
        self.init_states = init_states

        transforms = get_standardization(name)
        self.input_transform = transforms['input']
        self.output_transform = transforms['output']

        self.states_list = []
    
    def get_observation(self):
        states = self.get_eef_states()
        if self.use_standardization:
            states = self.input_transform(states)
        return {
            'states': states,
        }
    
    def do_action(self, actions):
        if self.use_standardization:
            actions = self.output_transform(actions)
        
        if self.control_mode == 'eef_absolute':
            self.set_eef_states(actions)
        elif self.control_mode == 'eef_delta_root':
            states = self.get_eef_states()
            # self.set_eef_states(delta_to_absolute_root_translation(actions, states))
            raise NotImplementedError
        elif self.control_mode == 'eef_delta_gripper':
            states = self.get_eef_states()
            states = delta_to_absolute_gripper_translation_align_piper(states, actions)
            self.set_eef_states(states)
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
    
    def get_eef_states(self):
        return self.states
    
    def set_eef_states(self, states):
        self.states = states
        self.states_list.append(states)

    def reset(self):
        self.states = self.init_states
    
    def stop(self):
        pass

    def visualize(self):
        positions = [state[0:3] for state in self.states_list]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        positions = np.array(positions)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.axis('equal')
        plt.show()


