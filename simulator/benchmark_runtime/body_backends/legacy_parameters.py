# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache-2.0
"""HA native controller parameters; preserved from the existing providers.

SONIC gains are the installed HA tuning, not unmodified upstream gains.
TWIST2 gains match HA G129_CFG_WITH_DEX3_WHOLEBODY, not MuJoCo deploy gains.
"""
import numpy as np


SONIC_ISAACLAB_JOINT_ORDER = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']

SONIC_DEFAULT_POS = np.array([-0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

G1_ACTION_SCALE_ISAACLAB = np.array([0.3506614566, 0.3506614566, 0.5475464463, 0.3506614566, 0.3506614566, 0.4385773242, 0.5475464463, 0.5475464463, 0.4385773242, 0.3506614566, 0.3506614566, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.4385773242, 0.0745008737, 0.0745008737, 0.0745008737, 0.0745008737], dtype=np.float32)

SONIC_PD_KP = np.array([99.02755, 99.02755, 40.168964, 99.02755, 99.02755, 28.445627, 40.168964, 40.168964, 28.445627, 99.02755, 99.02755, 14.222814, 14.222814, 48.445627, 48.445627, 34.222814, 34.222814, 28.445627, 28.445627, 24.222814, 24.222814, 19.222814, 19.222814, 14.222814, 14.222814, 16.77876, 16.77876, 16.77876, 16.77876], dtype=np.float32)

SONIC_PD_KD = np.array([3.1559467, 3.1559467, 1.2792531, 3.1559467, 3.1559467, 0.90733415, 1.2792531, 1.2792531, 0.90733415, 3.1559467, 3.1559467, 0.45366707, 0.45366707, 0.90733415, 0.90733415, 0.45366707, 0.45366707, 0.90733415, 0.90733415, 0.45366707, 0.45366707, 0.45366707, 0.45366707, 0.45366707, 0.45366707, 0.53407073, 0.53407073, 0.53407073, 0.53407073], dtype=np.float32)

SONIC_EFFORT_LIMIT = np.array([139.0, 139.0, 88.0, 139.0, 139.0, 25.0, 88.0, 88.0, 25.0, 139.0, 139.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

TWIST2_DEFAULT_POS = np.array([-0.2,0,0,0.4,-0.2,0]*2 + [0,0,0] + [0,0.4,0,1.2,0,0,0] + [0,-0.4,0,1.2,0,0,0], dtype=np.float32)
TWIST2_PD_KP = np.array([130,100,100,160,60,40]*2 + [150,150,150] + [40,40,40,40,20,20,20]*2, dtype=np.float32)
TWIST2_PD_KD = np.array([3,3,2,4.5,6.5,2]*2 + [4,4.5,8] + [5,5,5,5,1,1,1]*2, dtype=np.float32)
TWIST2_EFFORT_LIMIT = np.array([100,110,100,150,40,40]*2 + [150,150,150] + [60]*14, dtype=np.float32)
