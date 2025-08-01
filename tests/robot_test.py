import matplotlib.pyplot as plt
import sys
sys.path.append('.')
import draccus

from lerobot.scripts.server.configs import RobotClientConfig

from src.deploy.cameras.dummy import DummyCameraConfig
from src.deploy.robots.dummy import DummyConfig
from src.deploy.robots.utils import make_robot_from_config


@draccus.wrap()
def main(config: RobotClientConfig):
    robot = make_robot_from_config(config.robot)
    robot.connect()
    print(f"Robot connected: {robot.is_connected}")
    print("Observation features:", robot.observation_features)
    print("Action features:", robot.action_features)
    for x in range(100):
        robot.send_action({"x": x / 100, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "gripper": 0.5})
    for z in range(100):
        robot.send_action({"x": 1, "y": 0.0, "z": z / 100, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "gripper": 0.5})
    robot.disconnect()
    print("Robot disconnected.")
    robot.visualize()
    plt.show()

if __name__ == '__main__':
    main()