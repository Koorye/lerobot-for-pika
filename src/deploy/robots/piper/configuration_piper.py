from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    control_mode: str = 'eef_absolute'
    use_standardization: bool = True
    init_eef_state: list[int] = field(default_factory=lambda: [100000, 0, 300000, 0, 0, 0, 60000])