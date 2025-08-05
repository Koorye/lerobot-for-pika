import copy
from typing import Any

from lerobot.errors import DeviceNotConnectedError

from .piper import Piper
from .configuration_piper import PiperEndEffectorConfig
from ..misc import get_standardization, get_transform, get_visualizer


class PiperEndEffector(Piper):

    config_class = PiperEndEffectorConfig
    name = "piper_end_effector"

    def __init__(self, config: PiperEndEffectorConfig):
        super().__init__(config)

        self._base_state = None
        self._delta_with_previous = config.delta_with_previous

        self.standardization = get_standardization(self.name)
        self.transform = get_transform(config.control_mode, config.base_euler)
        self.visualizer = get_visualizer(config.init_ee_state, 'ee_absolute') if config.visualize else None
    
    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "dtype": "float32",
            "shape": (7,),
            "names": {
                "x": 0, "y": 1, "z": 2, 
                "roll": 3, "pitch": 4, "yaw": 5, 
                "gripper": 6,
            },
        }
    
    def connect(self):
        super().connect()
        self._base_state = self._get_ee_state()
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        state = self._get_ee_state() if self._delta_with_previous else copy.deepcopy(self._base_state)
        state = self.standardization.input_transform(state)

        action = self.transform(state, action)
        action = self.standardization.output_transform(action)

        self._set_ee_state(action)

        if self.visualizer:
            state = self.standardization.input_transform(self._get_ee_state())
            self.visualizer.add(state)
            self.visualizer.plot()
