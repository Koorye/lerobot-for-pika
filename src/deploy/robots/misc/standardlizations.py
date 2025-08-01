from abc import ABC, abstractmethod
import numpy as np


class BaseStandardization(ABC):
    @abstractmethod
    def input_transform(self, states):
        """
        Transform the input states to a standardized format.
        """
        pass

    @abstractmethod
    def output_transform(self, states):
        """
        Transform the output actions to a standardized format.
        """
        pass


class DummyStandardization(BaseStandardization):
    def input_transform(self, states):
        """
        Dummy standardization does not change the input states.
        """
        return states

    def output_transform(self, states):
        """
        Dummy standardization does not change the output actions.
        """
        return states


class PiperStandardization(BaseStandardization):
    def input_transform(self, states):
        """
        Convert Piper states to Pika format.
        """
        return [
            states[0] * 1e-6,  # x in meters
            states[1] * 1e-6,  # y in meters
            states[2] * 1e-6,  # z in meters
            states[3] * 1e-3 * np.pi / 180,  # rx in radians
            states[4] * 1e-3 * np.pi / 180,  # ry in radians
            states[5] * 1e-3 * np.pi / 180,  # rz in radians
            states[6] / 60000.0 * 1.6  # grip normalized to [0, 1.6]
        ]

    def output_transform(self, states):
        """
        Convert Pika actions back to Piper format.
        """
        return [
            int(states[0] * 1e6),  # x in 0.001mm
            int(states[1] * 1e6),  # y in 0.001mm
            int(states[2] * 1e6),  # z in 0.001mm
            int(states[3] * 180 / np.pi * 1e3),  # rx in 0.001 degree
            int(states[4] * 180 / np.pi * 1e3),  # ry in 0.001 degree
            int(states[5] * 180 / np.pi * 1e3),  # rz in 0.001 degree
            int(states[6] / 1.6 * 60000)   # grip in [0, 60000]
        ]


def get_standardization(standardization_type: str) -> BaseStandardization:
    """
    Factory function to get the standardization class based on the type.
    """
    if standardization_type == "dummy":
        return DummyStandardization()
    elif standardization_type == "piper":
        return PiperStandardization()
    else:
        raise ValueError(f"Unknown standardization type: {standardization_type}")