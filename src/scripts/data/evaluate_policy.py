"""
This script evaluates a policy on a dataset from the LeRobot repository.

Example command:
python src/scripts/data/evaluate_policy.py \
    --repo_id=lerobot/pika \
    --policy=act \
    --pretrained=path/to/pretrained/checkpoint \
    --device=cuda \
    --actions_per_chunk=16
"""

import sys
sys.path.append('.')

import argparse
import time
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.policies.factory import get_policy_class
from src.robots.misc import get_visualizer


@torch.no_grad()
def evaluate(args):
    dataset = LeRobotDataset(repo_id=args.repo_id)
    data_iterator = iter(dataset)

    policy_class = get_policy_class(args.policy)
    policy = policy_class.from_pretrained(args.pretrained)
    policy.to(args.device)

    first_sample = dataset[0]
    image_names = [key for key in first_sample.keys() if 'observation.images' in key]
    init_state = first_sample['observation.state'].numpy()

    assert len(init_state) in [7, 14], "Initial state should have 7 (for single arm) or 14 elements (for two arms)."

    if len(init_state) == 7:
        visualizer = get_visualizer(image_names, ['arm'], [init_state], 'ee_delta_gripper')
    else:
        visualizer = get_visualizer(image_names, ['arm_left', 'arm_right'], [init_state[:7], init_state[7:]], 'ee_delta_gripper')
    
    sample = next(data_iterator)
    prev_episode = sample['episode_index']

    while True:
        init_state = sample['observation.state'].numpy()
        if sample['episode_index'] != prev_episode:
            if len(init_state) == 7:
                visualizer.reset([init_state])
            else:
                visualizer.reset([init_state[:7], init_state[7:]])
            prev_episode = sample['episode_index']

        images = [sample[name].permute(1, 2, 0) for name in image_names]
        observation = {
            "observation.images.left_wrist_fisheye": sample['observation.images.left_wrist_fisheye'],
            "observation.images.right_wrist_fisheye": sample['observation.images.right_wrist_fisheye'],
            "observation.state": sample['observation.state'],
        }
        for key in observation:
            observation[key] = observation[key].unsqueeze(0).to(args.device)

        actions = policy.predict_action_chunk(observation).squeeze(0).cpu().numpy()[:args.actions_per_chunk]

        for action in actions:
            if len(init_state) == 7:
                visualizer.add(images, [action])
            else:
                visualizer.add(images, [action[:7], action[7:]])
            visualizer.plot()
        
        for _ in range(len(actions)):
            try:
                sample = next(data_iterator)
            except StopIteration:
                print("End of dataset reached.")
                return
            if sample['episode_index'] != prev_episode:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize dataset samples.")
    parser.add_argument('--repo_id', type=str, required=True, help='Repository ID of the dataset.')
    parser.add_argument('--policy', type=str, required=True, help='Policy type to use for evaluation.')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to the pretrained policy model.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the policy on (e.g., "cuda" or "cpu").')
    parser.add_argument('--actions_per_chunk', type=int, default=16, help='Number of actions to predict per chunk.')
    args = parser.parse_args()
    evaluate(args)
