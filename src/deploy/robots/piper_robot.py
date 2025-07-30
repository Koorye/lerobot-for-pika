import numpy as np
import time
from piper_sdk import C_PiperInterface_V2

from .dummy_robot import DummyRobot


class PiperRobot(DummyRobot):
    def __init__(
        self, 
        can, 
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.piper = C_PiperInterface_V2(can)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
    
    def get_eef_states(self):
        end_pose = self.piper.GetArmEndPoseMsgs().end_pose
        x, y, z, rx, ry, rz = end_pose.X_axis, end_pose.Y_axis, end_pose.Z_axis, \
                              end_pose.RX_axis, end_pose.RY_axis, end_pose.RZ_axis
        grip = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle
        print('get:', [x, y, z, rx, ry, rz, grip])
        return np.array([x, y, z, rx, ry, rz, grip])
    
    def set_eef_states(self, states):
        print('set:', states)
        self.piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        x, y, z, rx, ry, rz, grip = states[:7]
        x, y, z, rx, ry, rz, grip = int(x), int(y), int(z), int(rx), int(ry), int(rz), int(grip)
        self.piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        self.piper.GripperCtrl(grip, 1000, 0x01, 0)
        self.states_list.append(self.get_eef_states())
    
    def set_joint_states(self, states):
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        j1, j2, j3, j4, j5, j6 = states[:6]
        j1, j2, j3, j4, j5, j6 = int(j1), int(j2), int(j3), int(j4), int(j5), int(j6)
        self.piper.JointCtrl(j1, j2, j3, j4, j5, j6)

    def reset(self):
        self.set_eef_states(self.init_states)
        for _ in range(100):
            time.sleep(0.01)
        # self.set_joint_states(_INIT_JOINT_STATES)
        # for _ in range(100):
        #     time.sleep(0.01)
    
    def stop(self):
        while self.piper.DisablePiper():
            time.sleep(0.01)
