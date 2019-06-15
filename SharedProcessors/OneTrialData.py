from const import PROCESSED_DATA_PATH, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE, ROTATION_VIA_STATIC_CALIBRATION, \
    SPECIFIC_CALI_MATRIX, FILTER_BUFFER, FILTER_WIN_LEN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from StrikeOffDetectorIMU import StrikeOffDetectorIMUFilter


class OneTrialData:
    def __init__(self, subject_name, trial_name, sensor_sampling_fre, static_data_df=None):
        self._subject_name = subject_name
        self._trial_name = trial_name
        self._sensor_sampling_fre = sensor_sampling_fre
        self._static_data_df = static_data_df
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            self._side = 'l'       # 'l' or 'r'
            data_folder = '\\200Hz\\'
        else:
            self._side = 'r'       # 'l' or 'r'
            data_folder = '\\100Hz\\'
        # initialize the dataframe of gait data, including force plate, marker and IMU data
        gait_data_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + trial_name + '.csv'
        self.gait_data_df = pd.read_csv(gait_data_path, index_col=False)
        # initialize the dataframe of gait parameters, including loading rate, strike index, ...
        gait_param_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + 'param_of_' + trial_name + '.csv'

        buffer_sample_num = self._sensor_sampling_fre * FILTER_BUFFER
        self.gait_data_df = self.gait_data_df.loc[buffer_sample_num:, :]        # skip the first several hundred data
        if static_data_df is not None:
            self.gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            self.gait_param_df = self.gait_param_df.loc[buffer_sample_num:, :]

    def get_one_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]

        data = self.gait_data_df[column_names].values
        if ROTATION_VIA_STATIC_CALIBRATION:
            data_rotated = np.zeros(data.shape)
            if self._subject_name in SPECIFIC_CALI_MATRIX.keys() and\
                    IMU_location in SPECIFIC_CALI_MATRIX[self._subject_name].keys():
                    dcm_mat = SPECIFIC_CALI_MATRIX[self._subject_name][IMU_location]
            else:
                dcm_mat = self.get_rotation_via_static_cali(IMU_location)
            data_len = data.shape[0]
            for i_sample in range(data_len):
                if acc:
                    data_rotated[i_sample, 0:3] = np.matmul(dcm_mat, data[i_sample, 0:3])
                if gyr:
                    data_rotated[i_sample, 3:6] = np.matmul(dcm_mat, data[i_sample, 3:6])
                if mag:
                    data_rotated[i_sample, 6:9] = np.matmul(dcm_mat, data[i_sample, 6:9])
            return data_rotated
        else:
            return data

    def get_rotation_via_static_cali(self, IMU_location):
        axis_name_gravity = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        data_gravity = self._static_data_df[axis_name_gravity]
        vector_gravity = np.mean(data_gravity.values, axis=0)

        axis_name_mag = [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        try:
            data_mag = self._static_data_df[axis_name_mag]
        except KeyError:
            pass
        vector_mag = np.mean(data_mag.values, axis=0)

        vector_2 = vector_gravity / np.linalg.norm(vector_gravity)
        vector_0 = np.cross(vector_mag, vector_gravity)
        vector_0 = vector_0 / np.linalg.norm(vector_0)
        vector_1 = np.cross(vector_2, vector_0)
        vector_1 = vector_1 / np.linalg.norm(vector_1)

        dcm_mat = np.array([vector_0, vector_1, vector_2])
        return dcm_mat

    def get_grf_input_output(self, IMU_location, weight, from_IMU, acc=True, gyr=True, mag=False, grf_axis='z'):
        """
        GRFz: from strike to off
        acc and gyr: from off to off because information before strike might be useful
        """
        if from_IMU == 2:
            filter_delay = int(FILTER_WIN_LEN / 2)
        else:
            filter_delay = 0
        if not from_IMU:
            offs, step_num = self.get_offs()
            strikes, step_num = self.get_strikes()
        else:
            offs, strikes, step_num = self.get_offs_strikes_from_IMU(from_IMU)
        grf_data = - self.gait_data_df['f_1_' + grf_axis].values / weight
        IMU_data = self.get_one_IMU_data(IMU_location, acc, gyr, mag)
        step_grf_data, step_imu_data = [], []
        for i_step in range(step_num):
            strike_in_between = strikes[offs[i_step] < strikes]
            strike_in_between = strike_in_between[strike_in_between < offs[i_step+1]]
            if len(strike_in_between) != 1:
                continue
            # give IMU data a padding so that more information can be observed
            edge_extender = 4
            imu_start = strike_in_between[0] - edge_extender - filter_delay
            imu_end = offs[i_step + 1] + edge_extender - filter_delay
            grf_start = strike_in_between[0] - edge_extender - filter_delay
            grf_end = offs[i_step+1] + edge_extender - filter_delay

            # skip this step if the grf_end exceeds the maximum data length
            if grf_end > grf_data.shape[0]:
                continue

            step_imu_data.append(IMU_data[imu_start:imu_end, :])
            step_grf_data.append(grf_data[grf_start:grf_end])
        step_imu_data, step_grf_data = self.check_step_input_output(step_imu_data, step_grf_data)
        return step_imu_data, step_grf_data

    def get_off_to_off_input(self, IMU_location, acc=True, gyr=True, mag=False, from_IMU=True):
        """
        :param from_IMU: int, 0 for from force plate, 1 for from filtfilt, 2 for from lfilter
        :return: First three column, acc; second three column, gyr; seventh column, strike; eighth column, strike index
        """
        if not from_IMU:
            off_time_indexes, step_num = self.get_offs()
            strikes = self.gait_param_df[self._side + '_strikes']
        elif from_IMU == 1:
            off_time_indexes, strike_time_indexes, step_num = self.get_offs_strikes_from_IMU(from_IMU)
            strikes = self.gait_param_df['strikes_IMU']
        elif from_IMU == 2:
            off_time_indexes, strike_time_indexes, step_num = self.get_offs_strikes_from_IMU(from_IMU)
            strikes = self.gait_param_df['strikes_IMU_lfilter']

        IMU_data = self.get_one_IMU_data(IMU_location, acc, gyr, mag)
        strike_index = self.gait_param_df[self._side + '_strike_angle']     # !!! changes, just for testing
        return_data = np.column_stack([IMU_data, strikes, strike_index])
        step_data = []      # each element represent a step
        for i_step in range(step_num):
            step_start = off_time_indexes[i_step]
            step_end = off_time_indexes[i_step+1]
            step_data.append(return_data[step_start:step_end, :])
        step_data = self.check_step_data(step_data)
        return step_data

    def get_step_param(self, param_name, from_IMU=True):
        if from_IMU:
            offs, strikes, step_num = self.get_offs_strikes_from_IMU(from_IMU)
        else:
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

    def get_offs_strikes_from_IMU(self, from_IMU):
        """
        There is no side because by default 100Hz is the right side, 200Hz is the left side.
        :return:
        """
        if from_IMU == 1:
            off_column = 'offs_IMU'
            strike_column = 'strikes_IMU'
        elif from_IMU == 2:
            off_column = 'offs_IMU_lfilter'
            strike_column = 'strikes_IMU_lfilter'
        else:
            raise ValueError('Invalid from_IMU value. from_IMU: 0 for from plate, 1 for filtfilt, 2 for lfilter')
        offs = self.gait_param_df[off_column]
        offs = np.where(offs == 1)[0]
        step_num = len(offs) - 1

        strikes = self.gait_param_df[strike_column]
        strikes = np.where(strikes == 1)[0]
        return offs, strikes, step_num

    @staticmethod
    def check_step_data(step_data, up_diff_ratio=0.4, down_diff_ratio=0.3):        # check if step length is correct
        step_num = len(step_data)
        step_lens = np.zeros([step_num])
        for i_step in range(step_num):
            step_lens[i_step] = len(step_data[i_step])
        step_len_mean = np.mean(step_lens)
        acceptable_len_max = step_len_mean * (1+up_diff_ratio)
        acceptable_len_min = step_len_mean * (1-down_diff_ratio)
        step_data_new = []
        for i_step in range(step_num):
            if acceptable_len_min < step_lens[i_step] < acceptable_len_max:
                step_data_new.append(step_data[i_step])
        return step_data_new

    @staticmethod
    def check_step_input_output(step_input, step_output, up_diff_ratio=0.4, down_diff_ratio=0.3):
        step_num = len(step_input)
        step_lens = np.zeros([step_num])
        for i_step in range(step_num):
            step_lens[i_step] = len(step_input[i_step])
        step_len_mean = np.mean(step_lens)
        acceptable_len_max = step_len_mean * (1+up_diff_ratio)
        acceptable_len_min = step_len_mean * (1-down_diff_ratio)
        step_input_new, step_output_new = [], []
        for i_step in range(step_num):
            if acceptable_len_min < step_lens[i_step] < acceptable_len_max:
                step_input_new.append(step_input[i_step])
                step_output_new.append(step_output[i_step])
        return step_input_new, step_output_new


class OneTrialDataStatic(OneTrialData):
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


