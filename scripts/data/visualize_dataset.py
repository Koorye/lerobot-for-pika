import sys
sys.path.append('.')

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.deploy.robots.misc import get_visualizer


dataset = LeRobotDataset(repo_id='Koorye/pika')
first_sample = dataset[0]
image_names = [key for key in first_sample.keys() if 'observation.images' in key]
init_state = first_sample['observation.state']
visualizer = get_visualizer(image_names, ['left', 'right'], [init_state[:7], init_state[7:]], 'delta_gripper')

for sample in dataset:
    images = [sample[name].permute(1, 2, 0) for name in image_names]
    action = sample['action']
    visualizer.add(images, [action[:7], action[7:]])
    visualizer.plot()
