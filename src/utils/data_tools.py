import numpy as np


def eef_states_to_pos_rot_grip(states):
    assert len(states) % 7 == 0
    outs = []
    for i in range(0, len(states), 7):
        pos = states[i:i + 3]
        rot = states[i + 3:i + 6]
        grip = states[i + 6]
        outs.append((pos, rot, grip))
    return outs


def pos_rot_grip_to_eef_states(pos_rot_grip):
    outs = []
    for pos, rot, grip in pos_rot_grip:
        outs += list(pos) + list(rot) + [grip]
    return np.array(outs)