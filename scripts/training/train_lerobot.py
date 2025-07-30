import sys
sys.path.append('.')

import argparse
import importlib
import torch
import torch.nn.functional as F
from pathlib import Path

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features


def resize_with_pad(img, width, height, pad_value=-1):
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


def main(args):
    config = importlib.import_module('scripts.training.configs.' + args.config).TrainingConfig()

    output_directory = Path(config.training['checkpoint_dir'])
    device = torch.device(config.training['device'])

    dataset_metadata = LeRobotDatasetMetadata(config.data['repo_id'])
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() 
                       if ft.type is FeatureType.ACTION}
    input_keys = config.data['image_keys']
    if config.data['use_state']:
        input_keys.append("observation.state")
    input_features = {key: ft for key, ft in features.items() 
                      if key in input_keys and ft.type is not FeatureType.ACTION}

    model_type = config.model['type']
    del config.model['type']

    if model_type == 'diffusion':
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        cfg = DiffusionConfig(
            input_features=input_features, 
            output_features=output_features,
            **config.model,
        )
        policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)

    elif model_type == 'act':
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        cfg = ACTConfig(
            input_features=input_features, 
            output_features=output_features,
            **config.model,
        )
        policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)

    elif model_type == 'smolvla':
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        cfg = SmolVLAConfig(
            input_features=input_features, 
            output_features=output_features,
            **config.model,
        )
        policy = SmolVLAPolicy(cfg, dataset_stats=dataset_metadata.stats)

    else:
        raise NotImplementedError

    policy.train()
    policy.to(device)

    delta_timestamps = {
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    if cfg.observation_delta_indices is not None:
        for key in config.data['image_keys']:
            delta_timestamps[key] = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
        if config.data['use_state']:
            delta_timestamps["observation.state"] = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
    
    dataset = LeRobotDataset(config.data['repo_id'], delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training['batch_size'],
        num_workers=config.training['num_workers'],
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.training['learning_rate'])

    if config.training['load_step'] is not None:
        checkpoint_file = output_directory / f"step_{config.training['load_step']}"
        policy.from_pretrained(checkpoint_file)
        meta_file = output_directory / f"step_{config.training['load_step']}" / "meta.pt"
        with open(meta_file, 'r') as f:
            meta = torch.load(f)
        optimizer.load_state_dict(meta['optimizer'])
        step = meta['step']
        print(f"Loaded checkpoint from {checkpoint_file} at step {step}")
    else:
        step = 0

    # Run training loop.
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if config.data['resize_with_padding']:
                for key in config.data['image_keys']:
                    batch[key] = resize_with_pad(
                        batch[key], 
                        *config.data['resize_with_padding'], 
                    )

            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % config.training['log_frequency'] == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            
            if step > 0 and step % config.training['save_frequency'] == 0:
                output_file = output_directory / f"step_{step}"
                output_file.mkdir(parents=True, exist_ok=True)
                output_meta_file = output_directory / f"step_{step}" / "meta.pt"

                policy.save_pretrained(output_file)
                torch.save({
                    'step': step,
                    'optimizer': optimizer.state_dict(),
                }, output_meta_file)
                print(f"Saved checkpoint to {output_file}")

            step += 1
            
            if step >= config.training['num_training_steps']:
                done = True
                break

    output_file = output_directory / f"step_{step}"
    output_file.mkdir(parents=True, exist_ok=True)
    output_meta_file = output_directory / f"step_{step}" / "meta.pt"

    policy.save_pretrained(output_directory)
    torch.save({
        'step': step,
        'optimizer': optimizer.state_dict(),
    }, output_meta_file)
    print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LeRobot policy.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args)
