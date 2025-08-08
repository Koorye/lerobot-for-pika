import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataProcessorConfig:
    """
    Dataset generataion configuration class for the Pika dataset.

    Attributes:
        overwrite: If True, existing data in the specified root directory will be overwritten.
        check_only: If True, only checks the configuration without creating a dataset.
        source_data_roots: List of source data directories to process.
        image_height: Height of the camera frames.
        image_width: Width of the camera frames.
        rgb_dirs: List of directories containing RGB images.
        rgb_names: List of names for RGB images in the dataset.
        use_depth: If True, depth images will be included in the dataset.
        depth_dirs: List of directories containing depth images (if applicable).
        depth_names: List of names for depth images in the dataset (if applicable).
        action_name: Name of the action data field.
        action_dirs: List of directories containing action data.
        action_keys_list: List of action keys for each arm.
        position_nonoop_threshold: Threshold for considering a position action as a noop.
        rotation_nonoop_threshold: Threshold for considering a rotation action as a noop.
        gripper_nonoop_threshold: Threshold for considering a gripper action as a noop.
        transform_type: Type of transformation to apply to actions (e.g., 'absolute', 'delta_base', 'delta_gripper').
        use_state: If True, state information will be included in the dataset.
        state_name: Name of the state data field (if applicable).
        instruction_path: Path to the instruction file.
        default_instruction: Default instruction to use if none is provided.
        repo_id: Save repository ID for the dataset.
        data_root: Save root directory for storing the dataset.
        fps: Frames per second for the video.
        video_backend: Backend to use for video processing (e.g., 'opencv', 'ffmpeg').
    """

    overwrite: bool = True
    check_only: bool = False

    source_data_roots: List[str] = field(default_factory=lambda: [])

    image_height: int = 480
    image_width: int = 640
    rgb_dirs: List[str] = field(default_factory=lambda: [])
    rgb_names: List[str] = field(default_factory=lambda: [])

    use_depth: bool = False
    depth_dirs: List[str] = field(default_factory=lambda: [])
    depth_names: List[str] = field(default_factory=lambda: [])

    action_name: str = 'action'
    action_dirs: List[str] = field(default_factory=lambda: [])
    action_keys_list: List[List[str]] = field(default_factory=lambda: [[]])

    position_nonoop_threshold: float = 1e-3
    rotation_nonoop_threshold: float = math.radians(1.0)
    gripper_nonoop_threshold: float = 1e-2
    transform_type: str = 'ee_absolute'

    use_state: bool = False
    state_name: str = 'observation.state'

    instruction_path: str = 'instructions.json'
    default_instruction: str = 'do something'

    repo_id: str = 'lerobot/pika'
    data_root: Optional[str] = None
    fps: int = 30
    video_backend: str = 'pyav'

    def __post_init__(self):
        self.action_len = sum(len(keys) for keys in self.action_keys_list)


@dataclass
class RGBSingleArmDataProcessorConfig(DataProcessorConfig):
    """
    Configuration class for single-arm data processing.
    """

    rgb_dirs: List[str] = field(default_factory=lambda: [
        'camera/color/camera_realsense_c',
        'camera/color/pikaDepthCamera_l',
        'camera/color/pikaFisheyeCamera_l',
    ])
    rgb_names: List[str] = field(default_factory=lambda: [
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.left_wrist_fisheye',
    ])

    action_dirs: List[str] = field(default_factory=lambda: [
        'localization/pose/pika_l',
        'gripper/encoder/pika_l',
    ])
    action_keys_list: List[List[str]] = field(default_factory=lambda: [
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
    ])
    transform_type: str = 'ee_delta_gripper'


@dataclass
class RGBMultiArmDataProcessorConfig(DataProcessorConfig):
    """
    Configuration class for multi-arm data processing.
    """

    rgb_dirs: List[str] = field(default_factory=lambda: [
        'camera/color/camera_realsense_c',
        'camera/color/pikaDepthCamera_l',
        'camera/color/pikaFisheyeCamera_l',
        'camera/color/pikaDepthCamera_r',
        'camera/color/pikaFisheyeCamera_r',
    ])
    rgb_names: List[str] = field(default_factory=lambda: [
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.left_wrist_fisheye',
        'observation.images.right_wrist',
        'observation.images.right_wrist_fisheye',
    ])

    action_dirs: List[str] = field(default_factory=lambda: [
        'localization/pose/pika_l',
        'gripper/encoder/pika_l',
        'localization/pose/pika_r',
        'gripper/encoder/pika_r',
    ])
    action_keys_list: List[List[str]] = field(default_factory=lambda: [
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
    ])
    transform_type: str = 'ee_delta_gripper'


@dataclass
class RGBSingleArmDeltaGripperDataProcessorConfig(RGBSingleArmDataProcessorConfig):
    """
    Configuration class for single-arm data processing with delta gripper transformations.
    """

    transform_type: str = 'ee_delta_gripper'


@dataclass
class RGBMultiArmDeltaGripperDataProcessorConfig(RGBMultiArmDataProcessorConfig):
    """
    Configuration class for multi-arm data processing with delta gripper transformations.
    """

    transform_type: str = 'ee_delta_gripper'
