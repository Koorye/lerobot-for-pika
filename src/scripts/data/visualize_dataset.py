import sys
sys.path.append('.')

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.deploy.robots.misc import get_visualizer


dataset = LeRobotDataset(repo_id='Koorye/pika')
first_sample = dataset[0]
image_names = [key for key in first_sample.keys() if 'observation.images' in key]
init_state = first_sample['observation.state'].numpy()
visualizer = get_visualizer(image_names, ['left', 'right'], [init_state[:7], init_state[7:]], 'ee_delta_gripper')
prev_episode = None

for sample in dataset:
    if prev_episode is None:
        prev_episode = sample['episode_index']
    else:
        init_state = sample['observation.state'].numpy()
        if sample['episode_index'] != prev_episode:
            visualizer.reset([init_state[:7], init_state[7:]])
            prev_episode = sample['episode_index']

    images = [sample[name].permute(1, 2, 0) for name in image_names]
    action = sample['action'].numpy()
    visualizer.add(images, [action[:7], action[7:]])
    visualizer.plot()