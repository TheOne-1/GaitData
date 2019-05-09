
from const import PROCESSED_DATA_PATH, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from transforms3d.euler import euler2mat, quat2euler
import pickle


class TrialProcessor:
    def __init__(self, subject_name, trial_name, side, sensor_sampling_fre):
        self._subject_name = subject_name
        self._trial_name = trial_name
        self._side = side       # 'l' or 'r'
        self._sensor_sampling_fre = sensor_sampling_fre
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            data_folder = '\\200Hz\\'
        else:
            data_folder = '\\100Hz\\'
        # initialize the dataframe of gait data, including force plate, marker and IMU data
        gait_data_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + trial_name + '.csv'
        self.gait_data_df = pd.read_csv(gait_data_path, index_col=False)
        # initialize the dataframe of gait parameters, including loading rate, strike index, ...
        gait_param_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + 'param_of_' + trial_name + '.csv'
        self.gait_param_df = pd.read_csv(gait_param_path, index_col=False)
        # # initialize the legal steps
        # step_path = PROCESSED_DATA_PATH + subject_name + data_folder + 'step_' + side + '_of_' + trial_name + '.pkl'
        # self._legal_steps = pickle.load(step_path)

    def get_one_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]

        data = self.gait_data_df[column_names]
        return data

    # step_segmentation based on heel strike
    def get_step_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        offs, step_num = self.get_offs()
        data = self.get_one_IMU_data(IMU_location, acc, gyr, mag).values
        step_data = []      # each element represent a step
        for i_step in range(step_num):
            step_start = offs[i_step]
            step_end = offs[i_step+1]
            step_data.append(data[step_start:step_end, :])
        step_data = self.check_step_data(step_data)
        return step_data

    def get_step_param(self, param_name):
        offs, step_num = self.get_offs()
        column_name = self._side + '_' + param_name
        param_data = self.gait_param_df[column_name].values
        step_data = []      # each element represent a step
        for i_step in range(step_num):
            step_start = offs[i_step]
            step_end = offs[i_step+1]
            if len(param_data.shape) == 1:
                step_data.append(param_data[step_start:step_end])
            else:
                step_data.append(param_data[step_start:step_end, :])

        step_data = self.check_step_data(step_data)
        return step_data

    def get_strikes(self):
        strike_column = self._side + '_strikes'
        heel_strikes = self.gait_param_df[strike_column]
        strikes = np.where(heel_strikes == 1)[0]
        step_num = len(strikes) - 1
        return strikes, step_num

    def get_offs(self):
        off_column = self._side + '_offs'
        offs = self.gait_param_df[off_column]
        offs = np.where(offs == 1)[0]
        step_num = len(offs) - 1
        return offs, step_num

    @staticmethod
    def check_step_data(step_data, up_diff_ratio=0.5, down_diff_ratio=0.15):        # check if step length is correct
        step_num = len(step_data)
        step_lens = np.zeros([step_num])
        for i_step in range(step_num):
            step_lens[i_step] = len(step_data[i_step])
        step_len_mean = np.mean(step_lens)
        if step_num != 0:
            acceptable_len_max = step_len_mean * (1+up_diff_ratio)
        else:
            acceptable_len_max = 0
        acceptable_len_min = step_len_mean * (1-down_diff_ratio)

        step_data_new = []
        for i_step in range(step_num):
            if acceptable_len_min < step_lens[i_step] < acceptable_len_max:
                step_data_new.append(step_data[i_step])
        return step_data_new

    # # # get original Yaw angle (under no placement error condition)
    # @staticmethod
    # def get_original_yaw(IMU_location, side=None):
    #     if IMU_location is 'trunk':
    #         yaw = 0
    #     elif side is 'l':
    #         yaw = 0
    #     elif side is 'r':
    #         yaw = 180
    #     else:
    #         raise RuntimeError('Incorrect input variables')
    #     return yaw
    #
    # def get_cali_matrix(self):
    #     # get static trial data
    #     static_marker_file = PROCESSED_DATA_PATH + self._subject_name + '\\static.csv'
    #     data_static = pd.read_csv(static_marker_file)
    #     # get xsens orientation DCM
    #     xsens_quat_column_names = [self._side + '_' + self._IMU_location + '_quat_' + axis_name
    #                                for axis_name in ['w', 'x', 'y', 'z']]
    #     mean_quat_np = np.mean(data_static[xsens_quat_column_names])
    #     angles = np.rad2deg(quat2euler(mean_quat_np))
    #
    #     # get yaw_marker
    #     marker_column_names = [self._side + '_' + marker_name + '_' + axis_name for marker_name in ['toe', 'heel']
    #                            for axis_name in ['x', 'y', 'z']]
    #     data_static_marker = np.mean(data_static[marker_column_names])
    #     delta_x = -(data_static_marker[0] - data_static_marker[3])
    #     delta_y = data_static_marker[1] - data_static_marker[4]
    #     delta_z = data_static_marker[2] - data_static_marker[5]
    #     yaw_marker = np.rad2deg(np.arctan2(delta_x, delta_y))
    #     roll_marker = np.rad2deg(np.arctan2(delta_z, delta_y))
    #
    #     yaw_diff_degree = angles[2] - yaw_marker
    #     roll_diff_degree = angles[0] - roll_marker
    #     cali_matrix = euler2mat(np.deg2rad(roll_diff_degree), np.deg2rad(angles[1]),  np.deg2rad(yaw_diff_degree))
    #     return cali_matrix
    #
    # def get_cali_matrix_test(self):
    #     # get static trial data
    #     static_marker_file = PROCESSED_DATA_PATH + self._subject_name + '\\static.csv'
    #     data_static = pd.read_csv(static_marker_file)
    #     # get xsens orientation DCM
    #     xsens_quat_column_names = [self._side + '_' + self._IMU_location + '_quat_' + axis_name
    #                                for axis_name in ['w', 'x', 'y', 'z']]
    #     mean_quat_np = np.mean(data_static[xsens_quat_column_names])
    #     angles = np.rad2deg(quat2euler(mean_quat_np))
    #
    #     # get yaw_marker
    #     marker_column_names = [self._side + '_' + marker_name + '_' + axis_name for marker_name in ['toe', 'heel']
    #                            for axis_name in ['x', 'y', 'z']]
    #     data_static_marker = np.mean(data_static[marker_column_names])
    #     delta_x = -(data_static_marker[0] - data_static_marker[3])
    #     delta_y = data_static_marker[1] - data_static_marker[4]
    #     delta_z = data_static_marker[2] - data_static_marker[5]
    #     yaw_marker = np.rad2deg(np.arctan2(delta_x, delta_y))
    #     roll_marker = np.rad2deg(np.arctan2(delta_z, delta_y))
    #
    #     yaw_diff_degree = angles[2] - yaw_marker
    #     roll_diff_degree = (angles[0] - roll_marker)
    #     cali_matrix = euler2mat(np.deg2rad(roll_diff_degree), np.deg2rad(angles[1]),  np.deg2rad(yaw_diff_degree))
    #     return cali_matrix

    def prepare_IMU_data(self, trial_name, cali=False, filt=False):
        step_IMU_data = self.get_step_IMU_data(trial_name, acc=True, gyr=True)
        step_IMU_data_processed = []
        step_num = len(step_IMU_data)
        cali_matrix = self.get_cali_matrix()
        for i_step in range(step_num):
            IMU_data = step_IMU_data[i_step]
            if cali:
                IMU_data[:, 0:3] = np.matmul(cali_matrix, IMU_data[:, 0:3].T).T
                IMU_data[:, 3:6] = np.matmul(cali_matrix, IMU_data[:, 3:6].T).T
            if filt:
                for channel in range(6):
                    IMU_data[:, channel] = self.data_filt(IMU_data[:, channel], 20)
            step_IMU_data_processed.append(IMU_data)
        return step_IMU_data_processed













