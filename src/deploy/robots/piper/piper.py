import time
from functools import cached_property
from typing import Any

from lerobot.robots.robot import Robot

from piper_sdk import C_PiperInterface_V2

from .configuration_piper import PiperConfig


class PiperRobot(Robot):

    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.config = config
        self.arm = C_PiperInterface_V2(config.port)
    
    @cached_property
    def observation_features(self) -> dict:
        pass

    @cached_property
    def action_features(self) -> dict:
        pass

    @property
    def is_connected(self) -> bool:
        return self.arm.__connected
    
    def connect(self):
        self.arm.ConnectPort()
        while not self.arm.EnablePiper():
            print("Waiting for Piper to enable...")
            time.sleep(0.1)
    
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        pass

    def configure(self):
        pass
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        pass
    
    def disconnect(self):
        while self.arm.DisconnectPort():
            print("Waiting for Piper to disconnect...")
            time.sleep(0.1)
