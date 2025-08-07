import json
import numpy as np
import os
from collections import defaultdict

from .dummy_data_processor import DummyDataProcessor


def load_sync(file_path):
    with open(file_path, 'r') as f:
        filenames = f.readlines()
    return [filename.strip() for filename in filenames]


class PikaDataProcessor(DummyDataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self):
        for source_idx, source_data_root in enumerate(self.config.source_data_roots):
            episode_dirs = [d for d in os.listdir(source_data_root)]
            for episode_idx, episode_dir in enumerate(sorted(episode_dirs, key=lambda x: int(x[7:]))):
                episode_path = os.path.join(source_data_root, episode_dir)
                print(f'Processing source {source_idx + 1}/{len(self.config.source_data_roots)}, episode {episode_idx + 1}/{len(episode_dirs)}: {episode_path}')
                self._add_episode(episode_path)
        
    def _load_episode(self, episode_path):
        raw_images = defaultdict(list)
        for rgb_dir, rgb_name in zip(self.config.rgb_dirs, self.config.rgb_names):
            rgb_dir = os.path.join(episode_path, rgb_dir)
            
            if os.path.exists(os.path.join(rgb_dir, 'sync.txt')):
                filenames = load_sync(os.path.join(rgb_dir, 'sync.txt'))
            else:
                filenames = os.listdir(rgb_dir)
                filenames = [filename for filename in filenames if filename.endswith('.jpg') or filename.endswith('.png')]
                filenames.sort(key=lambda x: float(x[:-4]))

            for filename in filenames:
                image_path = os.path.join(rgb_dir, filename)
                raw_images[rgb_name].append(image_path)
            
        raw_actions = defaultdict(list)
        for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
            action_dir_ = os.path.join(episode_path, action_dir)

            if os.path.exists(os.path.join(action_dir_, 'sync.txt')):
                filenames = load_sync(os.path.join(action_dir_, 'sync.txt'))
            else:
                filenames = os.listdir(action_dir_)
                filenames = [filename for filename in filenames if filename.endswith('.json')]
                filenames.sort(key=lambda x: float(x[:-5]))

            for filename in filenames:
                action_path = os.path.join(action_dir_, filename)
                with open(action_path, 'r') as f:
                    action_data = json.load(f)
                action_data = np.array([action_data[key] for key in action_keys])
                raw_actions[action_dir].append(action_data)
        
        instruction_path = os.path.join(episode_path, self.config.instruction_path)
        with open(instruction_path, 'r') as f:
            instruction_data = json.load(f)
        
        instruction = instruction_data['instructions'][0]
        if instruction == 'null':
            instruction = self.config.default_instruction
        
        outputs = {
            'raw_images': raw_images,
            'raw_actions': raw_actions,
            'instruction': instruction
        }

        if self.config.use_depth:
            raw_depths = defaultdict(list)
            for depth_dir, depth_name in zip(self.config.depth_dirs, self.config.depth_names):
                depth_dir = os.path.join(episode_path, depth_dir)

                if os.path.exists(os.path.join(depth_dir, 'sync.txt')):
                    filenames = load_sync(os.path.join(depth_dir, 'sync.txt'))
                else:
                    filenames = os.listdir(depth_dir)
                    filenames = [filename for filename in filenames if filename.endswith('.png') or filename.endswith('.jpg')]
                    filenames.sort(key=lambda x: float(x[:-4]))
                
                for filename in filenames:
                    depth_path = os.path.join(depth_dir, filename)
                    raw_depths[depth_name].append(depth_path)
            outputs['raw_depths'] = raw_depths
        
        lens = []
        for rgb_name, images_list in raw_images.items():
            lens.append(len(images_list))
        for action_dir, actions_list in raw_actions.items():
            lens.append(len(actions_list))
        if self.config.use_depth:
            for depth_name, depth_list in raw_images.items():
                lens.append(len(depth_list))
        
        assert all(lens[0] == l for l in lens), "All lists must have the same length"
        
        return outputs
