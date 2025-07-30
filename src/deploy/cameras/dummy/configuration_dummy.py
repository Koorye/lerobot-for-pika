from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("dummy")
@dataclass
class DummyCameraConfig(CameraConfig):
    pass
