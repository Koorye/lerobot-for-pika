"""
This script visualizes samples from a dataset stored in a specified repository.

Example command:
python src/scripts/data/visualize_dataset.py --repo_id=lerobot/pika
"""

import argparse
import sys
sys.path.append('.')

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.robots.misc import get_visualizer


def visualize(args):
    dataset = LeRobotDataset(repo_id=args.repo_id)
    first_sample = dataset[0]
    image_names = [key for key in first_sample.keys() if 'observation.images' in key]
    init_state = first_sample['observation.state'].numpy()

    assert len(init_state) in [7, 14], "Initial state should have 7 (for single arm) or 14 elements (for two arms)."

    if len(init_state) == 7:
        visualizer = get_visualizer(image_names, ['arm'], [init_state], 'ee_delta_gripper')
    else:
        visualizer = get_visualizer(image_names, ['arm_left', 'arm_right'], [init_state[:7], init_state[7:]], 'ee_delta_gripper')
    
    prev_episode = None

    for sample in dataset:
        if prev_episode is None:
            prev_episode = sample['episode_index']
        else:
            init_state = sample['observation.state'].numpy()
            if sample['episode_index'] != prev_episode:
                if len(init_state) == 7:
                    visualizer.reset([init_state])
                else:
                    visualizer.reset([init_state[:7], init_state[7:]])
                
                prev_episode = sample['episode_index']

        images = [sample[name].permute(1, 2, 0) for name in image_names]
        action = sample['action'].numpy()

        if len(init_state) == 7:
            visualizer.add(images, [action])
        else:
            visualizer.add(images, [action[:7], action[7:]])
        
        visualizer.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize dataset samples.")
    parser.add_argument('--repo_id', type=str, required=True, help='Repository ID of the dataset.')
    args = parser.parse_args()
    visualize(args)
