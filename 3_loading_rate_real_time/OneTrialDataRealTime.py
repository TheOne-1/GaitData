"""
This is for real time processing
"""
from const import PROCESSED_DATA_PATH, HAISHENG_SENSOR_SAMPLE_RATE, ROTATION_VIA_STATIC_CALIBRATION, \
    SPECIFIC_CALI_MATRIX, TRIAL_START_BUFFER, MOCAP_SAMPLE_RATE
import numpy as np
import scipy.interpolate as interpo
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt


class OneTrialDataRealTime:
    def __init__(self, subject_name, trial_name, sensor_sampling_fre, static_data_df=None):
        self._subject_name = subject_name
        self._trial_name = trial_name
        self.inertial_input, self.aux_input, self.step_output = [], [], []

        self._sensor_sampling_fre = sensor_sampling_fre
        self.filter_win_len = 1 * sensor_sampling_fre
        self.filter_delay_estimation = int(self.filter_win_len / 2)

        self._static_data_df = static_data_df
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            self._side = 'l'  # 'l' or 'r'
            data_folder = '\\200Hz\\'
        else:
            self._side = 'r'  # 'l' or 'r'
            data_folder = '\\100Hz\\'
        # initialize the dataframe of gait data, including force plate, marker and IMU data
        gait_data_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + trial_name + '.csv'
        self.gait_data_df = pd.read_csv(gait_data_path, index_col=False)
        # initialize the dataframe of gait parameters, including loading rate, strike index, ...
        gait_param_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + 'param_of_' + trial_name + '.csv'
        if static_data_df is not None:
            self.gait_param_df = pd.read_csv(gait_param_path, index_col=False)

    # step_segmentation based on heel strike
    def get_trial_data(self, cut_off_fre_strike_off=10, cut_off_fre_input=99):
        """

        :return: First three column, acc; second three column, gyr; seventh column, strike; eighth column, strike index
        """
        strike_acc_width = 10
        strike_acc_prominence = 6
        strike_acc_height = -15
        off_gyr_thd = 4  # threshold the minimum peak of medio-lateral heel strike
        off_gyr_prominence = 2
        step_len_max = 100
        step_len_min = 50

        if self._sensor_sampling_fre == HAISHENG_SENSOR_SAMPLE_RATE:
            strike_delay, off_delay = 4, 4  # delay from the peak
            IMU_location = 'r_foot'
        elif self._sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            strike_delay, off_delay = 6, 8  # delay from the peak
            IMU_location = 'l_foot'
        else:
            raise ValueError('Wrong sensor sampling rate value')
        # strike_delay, off_delay = 0, 0  # set as zero for debugging

        wn_strike_off = cut_off_fre_strike_off / self._sensor_sampling_fre
        b_strike_off = signal.firwin(self.filter_win_len, wn_strike_off)
        zi_acc = signal.lfilter_zi(b_strike_off, 1)
        zi_gyr = signal.lfilter_zi(b_strike_off, 1)

        data_len = self.gait_data_df.shape[0]
        acc_z_negative = -self.gait_data_df[IMU_location + '_acc_z'].values
        acc_z_filtered = np.zeros([data_len])
        gyr_x_negative = -self.gait_data_df[IMU_location + '_gyr_x'].values
        gyr_x_filtered = np.zeros([data_len])

        wn_input = cut_off_fre_input / self._sensor_sampling_fre
        b_input = signal.firwin(self.filter_win_len, wn_input)
        zi_acc_gyr = []
        for i_channel in range(6):
            zi_acc_gyr.append(signal.lfilter_zi(b_input, 1))
        acc_gyr_filtered = np.zeros([data_len, 6])

        acc_gyr_data = self.__get_one_IMU_data(IMU_location)
        lr_data = self.__get_param('LR')

        # last_toe_off = None
        strike_list, off_list = [], []
        abandoned_step_num = 0
        trial_start_buffer_sample_num = int((1.5 + TRIAL_START_BUFFER) * self._sensor_sampling_fre)

        # find the first strike
        for i_sample in range(trial_start_buffer_sample_num):
            acc_z_filtered[i_sample], zi_acc = signal.lfilter(b_strike_off, 1, [acc_z_negative[i_sample]], zi=zi_acc)
            gyr_x_filtered[i_sample], zi_gyr = signal.lfilter(b_strike_off, 1, [gyr_x_negative[i_sample]], zi=zi_gyr)
            acc_gyr_filtered[i_sample, :], zi_acc_gyr = self.acc_gyr_filter(b_input, 1, acc_gyr_data[i_sample], zi_acc_gyr)
        peaks, _ = signal.find_peaks(gyr_x_filtered[:i_sample], height=off_gyr_thd,
                                     prominence=off_gyr_prominence)
        try:
            off_list.append(peaks[-1] + off_delay)
        except IndexError:
            plt.figure()
            plt.plot(gyr_x_filtered[:i_sample])
            plt.show()
            raise IndexError('Gyr peak not found')

        peaks, _ = signal.find_peaks(acc_z_filtered[:off_list[0]], width=strike_acc_width,
                                     prominence=strike_acc_prominence)
        try:
            strike_list.append(peaks[-1] + strike_delay)
        except IndexError:
            plt.figure()
            plt.plot(acc_z_filtered[:i_sample])
            plt.show()
            raise IndexError('Acc peak not found')

        # find strikes and offs in real time
        check_win_len = self.filter_win_len
        last_off = off_list[-1]
        for i_sample in range(i_sample + 1, data_len):
            acc_z_filtered[i_sample], zi_acc = signal.lfilter(b_strike_off, 1, [acc_z_negative[i_sample]], zi=zi_acc)
            gyr_x_filtered[i_sample], zi_gyr = signal.lfilter(b_strike_off, 1, [gyr_x_negative[i_sample]], zi=zi_gyr)
            acc_gyr_filtered[i_sample, :], zi_acc_gyr = self.acc_gyr_filter(b_input, 1, acc_gyr_data[i_sample], zi_acc_gyr)
            if i_sample - last_off > check_win_len:
                try:
                    acc_peak = self.find_peak_max(acc_z_filtered[last_off:i_sample], width=strike_acc_width,
                                                  prominence=strike_acc_prominence, height=strike_acc_height)
                    gyr_peak = self.find_peak_max(gyr_x_filtered[last_off:i_sample],
                                                  height=off_gyr_thd, prominence=off_gyr_prominence)
                except ValueError as e:
                    plt.figure()
                    plt.plot(acc_z_filtered[i_sample - check_win_len:i_sample])
                    plt.plot(gyr_x_filtered[i_sample - check_win_len:i_sample])
                    # if len(acc_peaks) == 0:
                    #     peak_name_str = 'acc'
                    # elif len(gyr_peaks) == 0:
                    #     peak_name_str = 'gyr'
                    # else:
                    #     raise IndexError(e)
                    # print('At sample {sample_num}, {name_str} peak not found, step abondand'.
                    #       format(sample_num=i_sample, name_str=peak_name_str))
                    plt.grid()
                    plt.show()
                    last_off = off_list[-1] + 70      # skip this step

                strike_list.append(acc_peak + last_off + strike_delay)
                off_list.append(gyr_peak + last_off + off_delay)
                last_off = off_list[-1]

                if step_len_min < off_list[-1] - off_list[-2] < step_len_max:
                    inertial_input, aux_input = self.convert_step_input(acc_gyr_filtered[off_list[-2]:off_list[-1], :],
                                                                        strike_list[-1] - off_list[-2])
                    self.inertial_input.append(inertial_input)
                    self.aux_input.append(aux_input)
                    the_output = self.convert_step_output(lr_data, off_list[-2], off_list[-1])
                    if the_output is None:
                        self.inertial_input.pop(-1)     # illegal lr detected
                        self.aux_input.pop(-1)
                    else:
                        self.step_output.append(np.array(the_output))
                else:
                    abandoned_step_num += 1

        # self.check_strike_off(acc_z_filtered, gyr_x_filtered, IMU_location, strike_list, off_list)
        print('Subject {sub}, trial {trial}, {num} steps abandoned.'.format(
            sub=self._subject_name, trial=self._trial_name, num=abandoned_step_num))

        return self.inertial_input, self.aux_input, self.step_output

    def acc_gyr_filter(self, b, a, sample_data, zi_acc_gyr):
        filtered_data = np.zeros([6])
        for i_channel in range(6):
            filtered_data[i_channel], zi_acc_gyr[i_channel] = signal.lfilter(
                b, a, [sample_data[i_channel]], zi=zi_acc_gyr[i_channel])
        return filtered_data, zi_acc_gyr

    def convert_step_input(self, step_acc_gyr_data, strike_sample_num):
        """
        CNN based algorithm improved
        """
        resample_len = 100
        data_clip_start, data_clip_end = 45, 75
        channel_num = step_acc_gyr_data.shape[1]
        step_input = np.zeros([data_clip_end-data_clip_start, channel_num])
        aux_input = np.zeros([2])
        for i_channel in range(channel_num):
            channel_resampled = OneTrialDataRealTime.resample_channel(step_acc_gyr_data[:, i_channel], resample_len)
            step_input[:, i_channel] = channel_resampled[data_clip_start:data_clip_end]
            aux_input[0] = step_acc_gyr_data.shape[0]
            aux_input[1] = strike_sample_num
        return step_input, aux_input

    def convert_step_output(self, lr_data, step_start, step_end):
        """
        Convert and clean the output at the same time
        :return:
        """
        step_lr = lr_data[step_start - self.filter_delay_estimation: step_end - self.filter_delay_estimation]
        lr_index = np.where(step_lr != 0)[0]
        if len(lr_index) != 1:
            return None
        else:
            return step_lr[lr_index]

    def find_peak_max(self, data_clip, height, width=None, prominence=None):
        """
        find the maximum peak
        :return:
        """
        peaks, properties = signal.find_peaks(data_clip, width=width, height=height, prominence=prominence)
        peak_heights = properties['peak_heights']
        max_index = np.argmax(peak_heights)
        return peaks[max_index]

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

    def check_strike_off(self, acc_z, gyr_x, IMU_location, estimated_strike_indexes, estimated_off_indexes):
        plt.figure()
        plt.title(self._trial_name + '   ' + IMU_location + '   acc_z')
        plt.plot(acc_z)
        strike_plt_handle_esti = plt.plot(estimated_strike_indexes, acc_z[estimated_strike_indexes], 'r*')
        off_plt_handle_esti = plt.plot(estimated_off_indexes, acc_z[estimated_off_indexes], 'rx')
        plt.grid()
        plt.legend([strike_plt_handle_esti[0], off_plt_handle_esti[0]], ['estimated_strikes', 'estimated_offs'])

        plt.figure()
        plt.title(self._trial_name + '   ' + IMU_location + '   gyr_x')
        plt.plot(gyr_x)
        strike_plt_handle_esti = plt.plot(estimated_strike_indexes, gyr_x[estimated_strike_indexes], 'r*')
        off_plt_handle_esti = plt.plot(estimated_off_indexes, gyr_x[estimated_off_indexes], 'rx')
        plt.grid()
        plt.legend([strike_plt_handle_esti[0], off_plt_handle_esti[0]], ['estimated_strikes', 'estimated_offs'])
        plt.show()

    # def get_step_param(self, param_name, from_IMU=True):

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

    def get_offs_strikes_from_IMU(self):
        """
        There is no side because by default 100Hz is the right side, 200Hz is the left side.
        :return:
        """
        off_column = 'offs_IMU'
        offs = self.gait_param_df[off_column]
        offs = np.where(offs == 1)[0]
        step_num = len(offs) - 1

        strike_column = 'strikes_IMU'
        strikes = self.gait_param_df[strike_column]
        strikes = np.where(strikes == 1)[0]
        return offs, strikes, step_num

    @staticmethod
    def check_step_data(step_data, up_diff_ratio=0.5, down_diff_ratio=0.15):  # check if step length is correct
        step_num = len(step_data)
        step_lens = np.zeros([step_num])
        for i_step in range(step_num):
            step_lens[i_step] = len(step_data[i_step])
        step_len_mean = np.mean(step_lens)
        if step_num != 0:
            acceptable_len_max = step_len_mean * (1 + up_diff_ratio)
        else:
            acceptable_len_max = 0
        acceptable_len_min = step_len_mean * (1 - down_diff_ratio)

        step_data_new = []
        for i_step in range(step_num):
            if acceptable_len_min < step_lens[i_step] < acceptable_len_max:
                step_data_new.append(step_data[i_step])
        return step_data_new

    @staticmethod
    def resample_channel(data_array, resampled_len):
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        data_len = data_array.shape[1]
        data_step = np.arange(0, data_len)
        resampled_step = np.linspace(0, data_len, resampled_len)
        tck, data_step = interpo.splprep(data_array, u=data_step, s=0)
        data_resampled = interpo.splev(resampled_step, tck, der=0)[0]
        return data_resampled

    @staticmethod
    def clean_aux_input(aux_input):
        """
        replace zeros by the average
        :param aux_input:
        :return:
        """
        aux_input_median = np.median(aux_input, axis=0)
        for i_channel in range(aux_input.shape[1]):
            zero_indexes = np.where(aux_input[:, i_channel] == 0)[0]
            aux_input[zero_indexes, i_channel] = aux_input_median[i_channel]
            if len(zero_indexes) != 0:
                print('Zero encountered in aux input. Replaced by the median')
        return aux_input

    def __get_one_IMU_data(self, IMU_location, acc=True, gyr=True, mag=False):
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
            if self._subject_name in SPECIFIC_CALI_MATRIX.keys() and \
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

    def __get_param(self, param_name):
        column_name = self._side + '_' + param_name
        param_data = self.gait_param_df[column_name].values
        return param_data


class OneTrialStatic(OneTrialDataRealTime):
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






