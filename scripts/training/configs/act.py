import torch
import torchvision.transforms as T
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    training = dict(
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_training_steps = 50000,
        batch_size = 64,
        num_workers = 16,
        learning_rate = 1e-4,
        checkpoint_dir = 'outputs/lerobot/act',
        log_frequency = 1,
        save_frequency = 1000,
        load_step = None,
    )

    data = dict(
        repo_id = "Koorye/pika-demo",
        image_keys = [
            "observation.images.cam_left_wrist_fisheye",
            "observation.images.cam_right_wrist_fisheye",
        ],
        resize_with_padding = (224, 224),
        use_state = False,
    )

    model = dict(
        type = 'act',
        n_obs_steps = 1,
        chunk_size = 16,
        n_action_steps = 16,
        push_to_hub = False,
    )
