import itertools
import os
import numpy as np
import shutil
from collections import defaultdict
from tqdm import tqdm

try:
    # v2.1
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.1'
except:
    # v2.0
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.0'

from .configuration_data_processor import DataProcessorConfig
from .misc.images import load_image
from .misc.transforms import get_transform


def get_lerobot_default_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


class DummyDataProcessor(object):
    """
    A dummy data processor for creating a LeRobot dataset with dummy data.
    This processor generates random images and actions, simulating a dataset for testing purposes.
    It supports both single-arm and dual-arm configurations, with options for depth images and state information.

    Attributes:
        config: DataProcessorConfig instance containing the configuration for the dataset generation.
                (seeing `src/data/configuration_data_processor.py` for details)
    
    Examples:
        ```python
        config = DataProcessorConfig(
            source_data_roots=['/path/to/source/data'],
            rgb_dirs=['camera/color/camera_realsense_c'],
            rgb_names=['observation.images.front'],
            action_dirs=['localization/pose/pika_l'],
            action_keys_list=[['x', 'y', 'z', 'roll', 'pitch', 'yaw']],
            use_depth=False,
            use_state=False,
            transform_type='ee_absolute',
            repo_id='lerobot/pika',
        )
        processor = DummyDataProcessor(config)
        # This will create a dataset with dummy data and save it to lerobot/pika in the default cache directory.
        processor.process_data()
        ```
    """

    def __init__(self, config: DataProcessorConfig):
        self.config = config

        if self.config.overwrite:
            if self.config.data_root is not None:
                data_root = self.config.data_root
                if os.path.exists(data_root):
                    print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                    if input().strip().lower() != 'y':
                        print('Exiting without overwriting.')
                        return
                    shutil.rmtree(data_root, ignore_errors=True)
            else:
                data_root = get_lerobot_default_root()
                data_root = os.path.join(data_root, self.config.repo_id)
                if os.path.exists(data_root):
                    print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                    if input().strip().lower() != 'y':
                        print('Exiting without overwriting.')
                        return
                    shutil.rmtree(data_root, ignore_errors=True)
        
        self.transform = get_transform(self.config.transform_type, self.config.action_len > 7)
        self.create_dataset()

    def create_dataset(self):
        if self.config.check_only:
            print('Check only mode, skipping dataset creation.')
            return
        
        rgb_config = {
            'dtype': 'video',
            'shape': (self.config.image_height, self.config.image_width, 3),
            'names': ['height', 'width', 'channels'],
        }
        features = {rgb_name: rgb_config for rgb_name in self.config.rgb_names}
        action_keys_flatten = list(itertools.chain.from_iterable(self.config.action_keys_list))
        features[self.config.action_name] = {
            'dtype': 'float64',
            'shape': (self.config.action_len,),
            'names': action_keys_flatten,
        }

        if self.config.use_state:
            features[self.config.state_name] = {
                'dtype': 'float64',
                'shape': (self.config.action_len,),
                'names': action_keys_flatten,
            }

        if self.config.use_depth:
            depth_config = {
                'dtype': 'uint16',
                'shape': (self.config.image_height, self.config.image_width),
                'name': ['height', 'width'],
            }
            for depth_name in self.config.depth_names:
                features[depth_name] = depth_config
        
        if self.config.data_root is not None:
            self.config.data_root = os.path.join(self.config.data_root, self.config.repo_id)
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            root=self.config.data_root,
            fps=self.config.fps,
            video_backend=self.config.video_backend,
            features=features,
        )
    
    def process_data(self):
        num_episodes = 3
        for episode_idx in range(num_episodes):
            print(f'Processing episode {episode_idx + 1}/{num_episodes}')
            self._add_episode('dummy')
    
    def _add_episode(self, episode_path):
        raw_outputs = self._load_episode(episode_path)
        
        if self.config.check_only:
            print(f'Check only mode, skipping adding episode {episode_path}')
            return

        raw_images = raw_outputs['raw_images']
        raw_actions = raw_outputs['raw_actions']
        instruction = raw_outputs['instruction']
        if self.config.use_depth:
            raw_depths = raw_outputs['raw_depths']
        
        indexs = list(range(len(raw_images[self.config.rgb_names[0]])))
        state = np.concatenate([raw_actions[action_dir][0] for action_dir in self.config.action_dirs])

        for i in tqdm(indexs[1:], desc=f'Adding episode {episode_path}'):
            next_state = np.concatenate([raw_actions[action_dir][i] for action_dir in self.config.action_dirs])
            if not self._check_nonoop_actions(state, next_state):
                print(f'Skipping frame {i} due to non-noop actions.')
                continue

            action = self.transform(state, next_state)

            frame = {rgb_name: load_image(raw_images[rgb_name][i]) 
                     for rgb_name in self.config.rgb_names}
            frame[self.config.action_name] = action.copy()

            if self.config.use_state:
                frame[self.config.state_name] = state.copy()

            if self.config.use_depth:
                frame.update({depth_name: load_image(raw_depths[depth_name][i]) 
                              for depth_name in self.config.depth_names})
            
            if _LEROBOT_VERSION == '2.0':
                self.dataset.add_frame(frame)
            elif _LEROBOT_VERSION == '2.1':
                self.dataset.add_frame(frame, task=instruction)
            else:
                raise ValueError(f'Unsupported LeRobot version: {_LEROBOT_VERSION}')

            state = next_state
            
        if _LEROBOT_VERSION == '2.0':
            self.dataset.save_episode(task=instruction)
        elif _LEROBOT_VERSION == '2.1':
            self.dataset.save_episode()
        else:
            raise ValueError(f'Unsupported LeRobot version: {_LEROBOT_VERSION}')
        
    def _load_episode(self, episode_path):
        num_frames_per_episode = 100

        raw_images = defaultdict(list)
        raw_actions = defaultdict(list)
        instruction = 'do something'
        
        for frame_idx in range(num_frames_per_episode):
            for rgb_name in self.config.rgb_names:
                image = np.random.randint(0, 255, (self.config.image_height, self.config.image_width, 3), dtype=np.uint8)
                raw_images[rgb_name].append(image)
            
            for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
                action_data = np.random.rand(len(action_keys))
                raw_actions[action_dir].append(action_data)
        
        outputs = {
            'raw_images': raw_images,
            'raw_actions': raw_actions,
            'instruction': instruction
        }

        if self.config.use_depth:
            raw_depths = defaultdict(list)
            for frame_idx in range(num_frames_per_episode):
                for depth_name in self.config.depth_names:
                    depth_image = np.random.randint(0, 65535, (self.config.image_height, self.config.image_width), dtype=np.uint16)
                    raw_depths[depth_name].append(depth_image)
            outputs['raw_depths'] = raw_depths
        
        return outputs
    
    def _check_nonoop_actions(self, states, actions):
        assert len(states) == len(actions), "States and actions must have the same length."

        for i in range(0, len(states), 7):
            position_diff = np.linalg.norm(states[i:i+3] - actions[i:i+3])
            rotation_diff = np.linalg.norm(states[i+3:i+6] - actions[i+3:i+6])
            gripper_diff = abs(states[i+6] - actions[i+6])
            if (
                position_diff > self.config.position_nonoop_threshold 
                or rotation_diff > self.config.rotation_nonoop_threshold
                or gripper_diff > self.config.gripper_nonoop_threshold
            ):
                return True
         
        return False