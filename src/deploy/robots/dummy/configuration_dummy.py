from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("dummy")
@dataclass
class DummyConfig(RobotConfig):
    pass
