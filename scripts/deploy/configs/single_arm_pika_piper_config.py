from dataclasses import dataclass

from src.deploy.cameras import PikaCamera
from src.deploy.robots import PiperRobot


@dataclass
class DeployConfig:
    host = '127.0.0.1'
    port = 8000
    frequency = 10
    prompt = 'do something'

    camera = PikaCamera(
        usb = '/dev/ttyUSB0',
        fisheye_camera_index = 0,
        realsense_serial_number = None,
        width = 640,
        height = 480,
        fps = 30,
        use_fisheye = True,
        use_realsense = False,
        use_realsense_depth = False,
        prefix = 'left_wrist',
    )

    robot = PiperRobot(
        can = 'can_left',
        control_mode = 'eef_delta_gripper',
        use_standardization = True,
        init_states = [100000, 0, 300000, 0, 90000, 0, 60000],
    )