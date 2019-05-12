import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# rotation with heel on the ground
data_path = 'D:\\Tian\\Research\\Projects\\HuaweiProject\\ExperimentProtocalData\\190310FunctionalCalibration\\'
vicon_path = data_path + 'vicon\\'
xsens_path = data_path + 'xsens\\'
xsens_name_start = 'MT_03700647_00'
xsens_name_end = '-000.txt'
acc_column_names = ['Acc_X', 'Acc_Y', 'Acc_Z']
gyr_column_names = ['Gyr_X', 'Gyr_Y', 'Gyr_Z']
mag_column_names = ['Mag_X', 'Mag_Y', 'Mag_Z']
marker_column_names = [marker + axis for marker in ['heel_', 'toe_', 'side_'] for axis in ['x', 'y', 'z']]
vicon_column_names = ['Frame', 'SubFrame'] + marker_column_names
static_start = 100
static_end = 400

for i_trial in range(3, 6):
    vicon_file_name = vicon_path + 'T' + str(i_trial) + '.csv'
    vicon_data = pd.read_csv(vicon_file_name, skiprows=[0, 1, 2, 4]).astype(float)
    vicon_data.columns = vicon_column_names
    vicon_data = vicon_data.drop(columns=['Frame', 'SubFrame'])
    vicon_data_mat = vicon_data.values
    vicon_data_mat = GYRSimulator.data_filt(vicon_data_mat, 10)

    center_marker = np.array([0, 0, 1])     # 没什么用
    R_standing_to_ground = GYRSimulator.get_segment_R()
    marker_cali_matrix = GYRSimulator.get_marker_cali_matrix(vicon_data_mat, static_start, static_end)  # 0 到 3 秒
    virtual_marker, R_IMU_transform = GYRSimulator.get_virtual_marker(center_marker, vicon_data_mat,
                                                                      marker_cali_matrix, R_standing_to_ground)
    gyr_simu = GYRSimulator.data_filt(GYRSimulator.get_gyr(vicon_data, R_IMU_transform), 20)
    gyr_simu_norm = norm(gyr_simu, axis=1)

    vector_gravity_simu = np.array([0, 0, 1])

    xsens_file_name = xsens_path + xsens_name_start + str(i_trial) + xsens_name_end
    xsens_data_raw = pd.read_csv(xsens_file_name, header=5, sep='\t').values
    # in xsens_data, acc: column 0 - 2, gyr: column 3 - 5, mag: column 6 - 8
    xsens_data = GYRSimulator.data_filt(xsens_data_raw[:, 2:11], 20)
    vector_gravity_xsens = np.mean(xsens_data[static_start:static_end, 0:3], axis=0)
    vector_mag_xsens = np.mean(xsens_data[static_start:static_end, 6:9])

    gyr_xsens = xsens_data[:, 3:6]
    gyr_xsens_norm = norm(gyr_xsens, axis=1)
    vicon_delay = GYRSimulator.sync_data(gyr_simu_norm, gyr_xsens_norm)
    # GYRSimulator.compare_gyr(gyr_simu_norm, gyr_xsens_norm, vicon_delay)
    euler_diff = GYRSimulator.get_orientation_diff(vector_gravity_xsens, gyr_xsens[vicon_delay:, :],
                                                   vector_gravity_simu, gyr_simu)

    plt.plot(euler_diff[:, 2])
    # print(np.mean(abs(euler_diff), axis=0))

    euler_from_xsens = np.mean(xsens_data_raw[static_start:static_end, 11:14], axis=0)
    print(euler_from_xsens)
plt.show()



# mag_rotated_all = []
# mag_original_all = []
# rotation_axes_all = []
# for ori_data in ori_data_all:
#     static_gravity = ori_data[acc_column_names].values[static_start:static_end, :]
#     vector_gravity = np.mean(static_gravity, axis=0)
#     vector_gravity = vector_gravity / norm(vector_gravity)
#     gyr_data = ori_data[gyr_column_names].values
#     gyr_normed = norm(gyr_data, axis=1)
#
#     gyr_peaks = find_peaks(gyr_normed, height=2, distance=50)[0]
#     if len(gyr_peaks) != 3:
#         plt.plot(gyr_normed)
#         plt.plot(gyr_peaks, gyr_normed[gyr_peaks], 'r.')
#         plt.show()
#         raise RuntimeError('Wrong peaks were found')
#
#     # uses the average of 4 samples before peaks
#     rotation_samples = [gyr_peak - i_sample for gyr_peak in gyr_peaks for i_sample in [5, 4, 3, 2]]
#     rotation_axes = normalize(gyr_data[rotation_samples])
#     vector_rotation = np.mean(rotation_axes, axis=0)
#     vector_foot = np.cross(vector_gravity, vector_rotation)
#
#     # dcm = np.array([[vector_rotation[0], vector_foot[0], vector_gravity[0]],
#     #                 [vector_rotation[1], vector_foot[1], vector_gravity[1]],
#     #                 [vector_rotation[2], vector_foot[2], vector_gravity[2]]])
#     dcm = np.array([vector_rotation,
#                     vector_foot,
#                     vector_gravity])
#
#     static_mag = normalize(ori_data[mag_column_names].values[static_start:static_end, :])
#     vector_mag = np.mean(static_mag, axis=0)
#
#     mag_rotated = np.matmul(dcm, vector_mag)
#     print(mag_rotated)
#     mag_rotated_all.append(mag_rotated)
#     mag_original_all.append(vector_mag)
#     rotation_axes_all.append(vector_rotation)
#
# # free rotation
