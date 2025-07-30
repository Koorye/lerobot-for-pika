import numpy as np
from dataclasses import dataclass


@dataclass
class CameraObservation:
    name: str
    rgb_image: np.ndarray
    depth_image: list[any] | None = None

    def to_dict(self):
        return {
            'rgb_names': self.rgb_names,
            'rgb_images': self.rgb_images,
            'depth_names': self.depth_names,
            'depth_images': self.depth_images,
        }
    
    @staticmethod
    def from_dict(data):
        return CameraObservation(
            rgb_names=data['rgb_names'],
            rgb_images=data['rgb_images'],
            depth_names=data.get('depth_names'),
            depth_images=data.get('depth_images'),
        )


@dataclass
class RobotObservation:
    eef_states: list[float] | None = None
    joint_states: list[float] | None = None