from dataclasses import dataclass

from src.deploy.cameras import DummyCamera, MultiCameras
from src.deploy.robots import PiperRobot, MultiRobots


@dataclass
class DeployConfig:
    host = '127.0.0.1'
    port = 8000
    frequency = 5
    prompt = 'do something'
    visualize_every = 50

    left_camera = DummyCamera(
        width = 640,
        height = 480,
        fps = 30,
        use_fisheye = True,
        use_realsense = False,
        use_realsense_depth = False,
        prefix = 'left_wrist',
    )
    right_camera = DummyCamera(
        width = 640,
        height = 480,
        fps = 30,
        use_fisheye = True,
        use_realsense = False,
        use_realsense_depth = False,
        prefix = 'right_wrist',
    )
    camera = MultiCameras(
        left_camera=left_camera,
        right_camera=right_camera,
    )

    left_robot = PiperRobot(
        can = 'can_left',
        control_mode = 'eef_delta_gripper',
        use_standardization = True,
        init_states = [100000, 0, 300000, 0, 90000, -30000, 60000],
    )
    right_robot = PiperRobot(
        can = 'can_right',
        control_mode = 'eef_delta_gripper',
        use_standardization = True,
        init_states = [100000, 0, 300000, 0, 90000, 30000, 60000],
    )
    robot = MultiRobots(
        left_robot=left_robot,
        right_robot=right_robot,
    )

