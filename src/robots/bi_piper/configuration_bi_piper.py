import numpy as np
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("bi_piper")
@dataclass
class BiPiperConfig(RobotConfig):
    port_left: str
    port_right: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    init_ee_state: list[int] = field(default_factory=lambda: [100000, 0, 300000, 0, 0, 0, 60000])


@RobotConfig.register_subclass("bi_piper_end_effector")
@dataclass
class BiPiperEndEffectorConfig(BiPiperConfig):
    control_mode: str = 'ee_absolute'
    delta_with_previous: bool = True
    base_euler: list[float] = field(default_factory=lambda: [0.0, 0.5 * np.pi, 0.0])
    visualize: bool = True
