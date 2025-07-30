import numpy as np

from ..deploy.utils.translations import deg_to_rad, rad_to_deg, deg001_to_deg, deg_to_deg001


def piper_to_pika(states):
    """
    Convert Piper states to Pika states.
    Params:
    - states: [x, y, z, rx, ry, rz, grip]
    x, y, z in 0.001mm
    rx, ry, rz in 0.001 degree
    grip in [0, 60000]
    Returns:
    - states: [x, y, z, rx, ry, rz, grip]
    x, y, z in meters
    rx, ry, rz in radians
    grip in [0, 1.6]
    """
    return np.array([
        states[0] * 1e-6,  # x in meters
        states[1] * 1e-6,  # y in meters
        states[2] * 1e-6,  # z in meters
        deg_to_rad(deg001_to_deg(states[3])),  # rx in radians
        deg_to_rad(deg001_to_deg(states[4])),  # ry in radians
        deg_to_rad(deg001_to_deg(states[5])),  # rz in radians
        states[6] / 60000.0 * 1.6  # grip normalized to [0, 1.6]
    ])


def pika_to_piper(states):
    """
    Convert Pika states to Piper states.
    Params:
    - states: [x, y, z, rx, ry, rz, grip]
    x, y, z in meters
    rx, ry, rz in radians
    grip in [0, 1.6]
    Returns:
    - states: [x, y, z, rx, ry, rz, grip]
    x, y, z in 0.001mm
    rx, ry, rz in 0.001 degree
    grip in [0, 60000]
    """
    return np.array([
        int(states[0] * 1e6),  # x in 0.001mm
        int(states[1] * 1e6),  # y in 0.001mm
        int(states[2] * 1e6),  # z in 0.001mm
        int(deg_to_deg001(rad_to_deg(states[3]))),  # rx in 0.001 degree
        int(deg_to_deg001(rad_to_deg(states[4]))),  # ry in 0.001 degree
        int(deg_to_deg001(rad_to_deg(states[5]))),  # rz in 0.001 degree
        int(states[6] / 1.6 * 60000)  # grip in [0, 60000]
    ])


_STANDARDIZATION = {
    'piper': {
        'input': piper_to_pika,
        'output': pika_to_piper,
    }
}


def get_standardization(name):
    if name in _STANDARDIZATION:
        return _STANDARDIZATION[name]
    else:
        raise ValueError(f"Unknown standardization name: {name}")
