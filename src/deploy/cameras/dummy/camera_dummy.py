import numpy as np
from typing import Any

from lerobot.cameras.camera import Camera


class DummyCamera(Camera):
    def __init__(self, config):
        super().__init__(config)
        self._is_connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self) -> None:
        print("Dummy camera connected")
        self._is_connected = True
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        raise NotImplementedError("DummyCamera does not support method find_cameras")

    def read(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        print("Dummy camera disconnected")
        self._is_connected = False

