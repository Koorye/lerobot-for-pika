import numpy as np
from typing import Any

from lerobot.cameras.camera import Camera


class DummyCamera(Camera):
    def __init__(self, config):
        super().__init__(config)
        self.is_connected = False

    def connect(self) -> None:
        print("Dummy camera connected")
        self.is_connected = True
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("DummyCamera does not support method find_cameras")

    def read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        print("Dummy camera disconnected")
        self.is_connected = False

