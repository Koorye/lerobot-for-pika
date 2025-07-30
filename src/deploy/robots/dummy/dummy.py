from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .configuration_dummy import DummyConfig


class DummyRobot(Robot):
    config_class = DummyConfig
    name = "dummy"

    def __init__(self, config: DummyConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.is_connected = False
    
    @cached_property
    def observation_features(self) -> dict:
        motor_features = {}

        cameras_features = {}
        for cam_name, cam in self.cameras.items():
            if hasattr(cam, 'observation_features'):
                features = cam.observation_features
                cameras_features.update({f"{cam_name}.{k}": v for k, v in features.items()})
            else:
                cameras_features[cam_name] = (cam.height, cam.width, 3)
        
        return {**motor_features, **cameras_features}

    @cached_property
    def action_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.is_connected and all(self.camera.is_connected for self.camera in self.cameras.values())
    
    def connect(self):
        print("Dummy robot connected")
        self.is_connected = True
    
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        pass

    def configure(self):
        pass
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        print(action)
    
    def disconnect(self):
        print("Dummy robot disconnected")
        self.is_connected = False
        for camera in self.cameras.values():
            camera.disconnect()
