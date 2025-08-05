import numpy as np
from dataclasses import dataclass,field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    standardize: bool = True
    control_mode: str = 'ee_absolute'
    init_ee_state: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0.5 * np.pi, 0, 0])
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.5 * np.pi, 0.0])
    visualize: bool = True