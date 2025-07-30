import matplotlib.pyplot as plt
import numpy as np


class MultiRobots(object):
    def __init__(
        self, 
        left_robot,
        right_robot,
    ):
        self.left_robot = left_robot
        self.right_robot = right_robot
    
    def get_observation(self):
        left_obs = self.left_robot.get_observation()
        right_obs = self.right_robot.get_observation()
        states = np.concatenate((left_obs['states'], right_obs['states']))
        return {
            'states': states,
        }
    
    def do_action(self, action):
        left_action = action[:7]
        right_action = action[7:]
        self.left_robot.do_action(left_action)
        self.right_robot.do_action(right_action)
    
    def get_eef_state(self):
        left_state = self.left_robot.get_eef_state()
        right_state = self.right_robot.get_eef_state()
        return np.concatenate((left_state, right_state))
    
    def set_eef_state(self, action):
        left_action = action[:7]
        right_action = action[7:]
        self.left_robot.set_eef_state(left_action)
        self.right_robot.set_eef_state(right_action)
    
    def reset(self):
        self.left_robot.reset()
        self.right_robot.reset()

    def stop(self):
        self.left_robot.stop()
        self.right_robot.stop()

    def visualize(self):
        left_positions = [state[0:3] for state in self.left_robot.states_list]
        right_positions = [state[0:3] for state in self.right_robot.states_list]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        left_positions = np.array(left_positions)
        ax.scatter(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], c='r', marker='o')
        right_positions = np.array(right_positions)
        ax.scatter(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.axis('equal')
        plt.show()