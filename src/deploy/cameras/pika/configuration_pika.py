from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("pika")
@dataclass
class PikaCameraConfig(CameraConfig):
    usb: str
    fisheye_camera_index: int | None = None
    realsense_serial_number: str | None = None

    def __post_init__(self):
        if self.fisheye_camera_index is None and self.realsense_serial_number is not None:
            raise ValueError("One of fisheye camera index or realsense serial number must be provided")
