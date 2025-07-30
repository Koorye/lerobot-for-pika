from dataclasses import dataclass

from src.deploy.cameras import DummyCamera
from src.deploy.robots import DummyRobot


@dataclass
class DeployConfig:
    host = '127.0.0.1'
    port = 8000
    frequency = 10000
    prompt = 'do'

    camera = DummyCamera(
        width = 640,
        height = 480,
        fps = 30,
        use_fisheye = True,
        use_realsense = False,
        use_realsense_depth = False,
        prefix = 'left_wrist',
    )

    robot = DummyRobot(
        control_mode = 'eef_delta_gripper',
        use_standardization = True,
        init_states = [100000, 0, 300000, 0, 0, 0, 60000],
    )