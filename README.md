# Pi-0 For Pika

A pipeline for Data Processing, Model Training, and Real-world Deployment for Pika.

## Installation

```bash
git clone https://github.com/Koorye/pi0-for-pika.git
cd pi0-for-pika/scripts/installation
conda create -n pi0-for-pika python=3.11
conda activate pi0-for-pika
bash install.sh
```

After that, test GPUs by running: 
```bash
python scripts/installation/check_gpu_available.py
```

If all GPUs are available, this script will return:
```
GPU is available for PyTorch, TensorFlow, and JAX.
```

## Usage

### Data Processing

1. Edit or Add the configuration file in `scripts/data/configs` to set your parameters:
   - `source_data_roots`: The root directory of the source Pika dataset.
   - `image_height`, `image_width`, `rgb_dirs`, `rgb_names`: The image dimensions and directories for RGB images.
   - `depth_dirs`, `depth_names`: The directories for depth images.
   - `action_name`, `action_dirs`, `action_keys_list`: The action names and directories for actions.
   - `repo_id`, `data_root`, `fps`, `video_backend`: Several parameters for the dataset format.

Single-arm and multi-arm data processing config examples are available in `scripts/data/configs/single_arm.yaml` and `scripts/data/configs/multi_arm.yaml`, respectively.

2. Run the data processing script to prepare the dataset:
```bash
python scripts/data/pika2lerobot.py --config single_arm
# or
python scripts/data/pika2lerobot.py --config multi_arm
```

3. After running the script, the processed lerobot dataset will be saved in default lerbot dataset directory or the directory specified in the config file under `data_root`, which will have the following structure:
```bash
pika-example
├── data
│   └── chunk-000
│       └── episode_000000.parquet
│       └── ...
├── meta
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── info.json
│   └── tasks.jsonl
└── videos
    └── chunk-000
        ├── observation.images.cam_left_wrist
        │   └── episode_000000.mp4
        │   └── ...
        ├── observation.images.cam_left_wrist_fisheye
        │   └── episode_000000.mp4
        │   └── ...
        ├── observation.images.cam_right_wrist
        │   └── episode_000000.mp4
        │   └── ...
        ├── observation.images.cam_right_wrist_fisheye
        │   └── episode_000000.mp4
        │   └── ...
        └── observation.images.cam_third
            └── episode_000000.mp4
        │   └── ...
```

### Model Training

The following models are available for training:
- Diffusion Policy
- ACT (Action Chunking with Transformers)
- SmolVLA
- Pi-0

#### Training Diffusion Policy, ACT, and SmolVLA

The config for training these models are available in `scripts/training/configs`, and you can find the example configs in:
- `scripts/training/configs/act.py`
- `scripts/training/configs/smolvla.py`
- `scripts/training/configs/diffusion_policy.py`

Each config file contains the following sections:
- `training`: Configuration for the training process, including device, batch size, learning rate, training steps, checkpoint, etc.
- `data`: Configuration for the dataset, including repo id, image keys, state keys, resize configurations, etc.
- `model`: Configuration for the model architecture and training parameters, seeing the specific model documentation in lerobot for details.

My training hyperparameters:
| Hyperparameter | Value |
|----------------|-------|
| Batch Size     | 64    |
| Training Steps | 50000 |
| Learning Rate  | 1e-4 |
| Chunk Size     | 16   |
| Resize Shape | 84 for DP, 224 for ACT, 512 for SmolVLA |
| Observation Steps | 1 for ACT, 2 for DP and SmolVLA |

⚠️ One GPU with at least 12GB memory is enough to run the training for these models.

To train the models, you can run the training script with the specified config:
```bash
python scripts/training/train_lerobot.py --config diffusion
# or
python scripts/training/train_lerobot.py --config act
# or
python scripts/training/train_lerobot.py --config smolvla
```

#### Training Pi-0

The pika config has been added to `src/training/openpi/src/openpi/training/config.py`, available configs include:

- `pi0_pika_lora`: fine-tuning with LoRA
- `pi0_pika`: full fine-tuning

Training hyperparameters:
| Hyperparameter | Value |
|----------------|-------|
| Batch Size     | 32    |
| Training Steps | 30000 |
| Learning Rate  | 2.5e-5 |

⚠️ State token is replaced by a zero vector in the model, which is different from the original Pi-0 paper. 
This is due to the fact that the state token is not available in the Pika dataset. 
See the modification in line 217 of `src/training/openpi/src/openpi/models/pi0.py`.

⚠️ You need one GPU with at least 24GB memory to run the training.

1. Compute the dataset statistics:
```bash
python scripts/training/compute_norm_stats.py --config-name your_config
```

2. Train the model using the training script:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/training/train.py your_config --exp-name=your_experiment --overwrite
```

(Optional) If you want to add your own training config, please follow the format in:

- `src/training/openpi/src/openpi/training/config.py`
- `src/training/openpi/src/openpi/policies`

and add your config to the `_CONFIGS` list.

### Real-world Deployment

#### Deploy Diffusion Policy, ACT, and SmolVLA

TODO

#### Deploy Pi-0

1. Deploy model server:
```bash
python scripts/deploy/run_server.py \
    --config your_training_config \
    --checkpoint your_model_checkpoint \
    --port 8000
```

2. Connect usb to your robot arm and check all cans are available:
```bash
bash scripts/deploy/find_all_can_port.sh
```
The result will be similar to:
```bash
Both ethtool and can-utils are installed.
Interface can0 is connected to USB port 3-1.4:1.0
Interface can1 is connected to USB port 3-1.1:1.0 # for multi-arm setup, if you have two arms connected
```

3. Edit `scripts/deploy/activations/can_muti_activate.sh` to update the USB_PORTS variable with the ports found in the previous step. For example:
```bash
USB_PORTS["3-1.4:1.0"]="can_left:1000000"
USB_PORTS["3-1.1:1.0"]="can_right:1000000" # for multi-arm setup, if you have two arms connected
```
Then run the script to activate the CAN interfaces:
```bash
bash scripts/deploy/activations/can_activate.sh can_piper 1000000 "3-1.4:1.0" # for single-arm setup
# or
bash scripts/deploy/activations/can_muti_activate.sh # for multi-arm setup
```

4. Run `ifconfig` to check if all can interfaces are available.

5. Connect usb to Pika and check if the can interfaces are available:
```bash
python deploy/activations/multi_device_detector.py
```
And make sure each usb and fisheye camera index in the config file matches the actual camera.

6. Run the client to interact with the model server:

```bash
python scripts/deploy/run_client.py --config single_arm_pika_piper_config
# or
python scripts/deploy/run_client.py --config multi_arm_pika_piper_config
```