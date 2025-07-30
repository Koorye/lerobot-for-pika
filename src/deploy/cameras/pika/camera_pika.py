import numpy as np
import time
from typing import Any

from lerobot.cameras.camera import Camera

from pika import sense

from .configuration_pika import PikaCameraConfig


class PikaCamera(Camera):
    def __init__(self, config: PikaCameraConfig):
        super().__init__(config)
        self.usb = config.usb
        self.fisheye_camera_index = config.fisheye_camera_index
        self.realsense_serial_number = config.realsense_serial_number
        self.sense = sense(self.usb)

    @property
    def is_connected(self) -> bool:
        return self.sense.is_connected
    
    @property
    def observation_features(self) -> dict[str, tuple]:
        cameras = {}
        if self.fisheye_camera_index is not None:
            cameras['fisheye'] = (self.height, self.width, 3)
        if self.realsense_serial_number is not None:
            cameras['realsense'] = (self.height, self.width, 3)
            cameras['depth'] = (self.height, self.width)
        return cameras

    def connect(self) -> None:
        while not self.sense.connect():
            print("Waiting for Pika camera to connect...")
            time.sleep(0.1)

        self.sense.set_camera_param(self.width, self.height, self.fps)

        if self.fisheye_camera_index is not None:
            self.sense.set_fisheye_camera_index(self.fisheye_camera_index)
            self.fisheye_camera = self.sense.get_fisheye_camera()

        if self.realsense_serial_number is not None:
            self.sense.set_realsense_serial_number(self.realsense_serial_number)
            self.realsense_camera = self.sense.get_realsense_camera()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("PikaCamera does not support method find_cameras")

    def read(self) -> np.ndarray:
        outputs = {}

        if self.fisheye_camera_index is not None:
            success, frame = self.fisheye_camera.get_frame()
            if not success:
                raise RuntimeError("Failed to get fisheye frame")
            outputs['fisheye'] = frame[:, :, ::-1]
        
        if self.realsense_serial_number is not None:
            success, color_frame = self.realsense_camera.get_color_frame()
            if not success:
                raise RuntimeError("Failed to get realsense color frame")
            outputs['realsense'] = color_frame[:, :, ::-1]
            
            success, depth_frame = self.realsense_camera.get_depth_frame()
            if not success:
                raise RuntimeError("Failed to get realsense depth frame")
            outputs['depth'] = depth_frame
        
        return outputs

    def disconnect(self) -> None:
        self.sense.disconnect()
