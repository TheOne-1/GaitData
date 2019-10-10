from const import SUB_NAMES, MOCAP_SAMPLE_RATE
import numpy as np
import pandas as pd
from transforms3d.euler import mat2euler
from StrikeOffDetectorIMU import StrikeOffDetectorIMU
from OneTrialData import OneTrialDataStatic


def get_rotation_via_acc_mag_cali(static_data_df, segment):
    axis_name_gravity = [segment + '_acc_' + axis for axis in ['x', 'y', 'z']]
    data_gravity = static_data_df[axis_name_gravity]
    data_gravity = StrikeOffDetectorIMU.data_filt(data_gravity, 2, MOCAP_SAMPLE_RATE)
    vector_gravity = np.mean(data_gravity, axis=0)

    axis_name_mag = [segment + '_mag_' + axis for axis in ['x', 'y', 'z']]
    data_mag = static_data_df[axis_name_mag]
    data_mag = StrikeOffDetectorIMU.data_filt(data_mag, 2, MOCAP_SAMPLE_RATE)
    vector_mag = np.mean(data_mag, axis=0)

    vector_2 = vector_gravity / np.linalg.norm(vector_gravity)
    vector_0 = np.cross(vector_mag, vector_gravity)
    vector_0 = vector_0 / np.linalg.norm(vector_0)
    vector_1 = np.cross(vector_2, vector_0)
    vector_1 = vector_1 / np.linalg.norm(vector_1)

    dcm_mat = np.array([vector_0, vector_1, vector_2])
    euler_angles = np.round(np.rad2deg(mat2euler(dcm_mat)), 3)
    # """ Test """
    # if self._trial_name in ['nike SI 24', 'mini SI 24']:
    #     dcm_mat_sawp = np.swapaxes(dcm_mat, 0, 1)        # take the transpose of rotation matrix
    #     euler_angles = mat2euler(dcm_mat_sawp)
    #     euler_angles = [round(np.rad2deg(angle), 2) for angle in euler_angles]
    #     print(IMU_location + ' ' + str(euler_angles), end='')
    #     if IMU_location == 'l_foot':
    #         print()
    return euler_angles


np.set_printoptions(suppress=True)
for segment in ['l_shank', 'l_foot']:
    print(segment)
    for sub_name in SUB_NAMES:
        print(sub_name[:10], end='\t')
        # for trial in ['nike static', 'mini static']:
        nike_static_trial = OneTrialDataStatic(sub_name, 'nike static', 200)
        nike_static_df = nike_static_trial.get_multi_IMU_data_df([segment], acc=True, mag=True)
        nike_euler_angles = get_rotation_via_acc_mag_cali(nike_static_df, segment)

        mini_static_trial = OneTrialDataStatic(sub_name, 'mini static', 200)
        mini_static_df = mini_static_trial.get_multi_IMU_data_df([segment], acc=True, mag=True)
        mini_euler_angles = get_rotation_via_acc_mag_cali(mini_static_df, segment)

        print(mini_euler_angles - nike_euler_angles)












