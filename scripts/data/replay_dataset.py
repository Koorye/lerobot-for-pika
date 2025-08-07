import sys
sys.path.append('.')

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.deploy.robots.misc import get_standardization
from src.deploy.robots import BiPiperEndEffector, BiPiperEndEffectorConfig, make_robot_from_config


dataset = LeRobotDataset(repo_id='Koorye/pika')

first_sample = dataset[0]
init_state = first_sample['observation.state'][:7]
standradization = get_standardization('piper')
init_state = standradization.output_transform(init_state)

robot_config = BiPiperEndEffectorConfig(
    port_left='can_left',
    port_right='can_right',
    cameras={},
    init_ee_state=[100000, 0, 300000] + init_state[3:7],
    control_mode='ee_delta_gripper',
)
robot = make_robot_from_config(robot_config)

for sample in dataset:
    action = sample['action']
    robot.send_action(action)
