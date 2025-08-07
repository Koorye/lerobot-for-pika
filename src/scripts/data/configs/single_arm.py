import math
from dataclasses import dataclass


@dataclass
class DataConfig:
    overwrite = True
    check_only = False

    source_data_roots = [
        'examples/pika_example_data',
    ]

    image_height = 480
    image_width = 640
    rgb_dirs = [
        'camera/color/camera_realsense_c',
        'camera/color/pikaDepthCamera_l',
        'camera/color/pikaFisheyeCamera_l',
    ]
    rgb_names = [
        'observation.images.cam_third',
        'observation.images.cam_left_wrist',
        'observation.images.cam_left_wrist_fisheye',
    ]

    use_depth = True
    depth_dirs = [
        'camera/depth/pikaDepthCamera_c',
        'camera/depth/pikaDepthCamera_l',
    ]
    depth_names = [
        'observation.depths.cam_third',
        'observation.depths.cam_left_wrist',
    ]

    action_name = 'action'
    action_dirs = [
        'localization/pose/pika_l',
        'gripper/encoder/pika_l',
    ]
    action_keys_list = [
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
    ]
    position_nonoop_threshold = 1e-2 # 1cm
    rotation_nonoop_threshold = math.pi / 180 # 1 degree
    gripper_nonoop_threshold = 1e-2 # 1mm
    transform_type = 'ee_delta_gripper'

    use_state = True
    state_name = 'observation.state'

    instruction_path = 'instructions.json'
    default_instruction = 'do something'

    repo_id = 'Koorye/pika-example'
    data_root = None
    fps = 10
    video_backend = 'pyav'
