import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
from const import PROCESSED_DATA_PATH, FILE_NAMES, PLATE_SAMPLE_RATE, MOCAP_SAMPLE_RATE, HAISHENG_SENSOR_SAMPLE_RATE, \
    LOADING_RATE_NORMALIZATION
import scipy.interpolate as interpo
import xlrd
from scipy.signal import butter, filtfilt, find_peaks
from numpy.linalg import norm


class ParamProcessor:
    def __init__(self, path, sub_name, readme_xls):
        self._sub_name = sub_name
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        self.__weight = readme_sheet.cell_value(17, 1)  # in kilos
        self.__height = readme_sheet.cell_value(18, 1)  # in meters
        # self.__path_fre = path + '{frequency}\\'

        fre_100_path = path + '\\' + sub_name + '\\100Hz\\'
        fre_200_path = path + '\\' + sub_name + '\\200Hz\\'
        fre_1000_path = path + '\\' + sub_name + '\\1000Hz\\'
        static_data_100_df = pd.read_csv(fre_100_path + FILE_NAMES[0] + '.csv', index_col=False)
        static_data_200_df = pd.read_csv(fre_200_path + FILE_NAMES[0] + '.csv', index_col=False)
        for trial_name in FILE_NAMES[3:]:
            # initialize 100 Hz parameter
            self._current_trial = trial_name
            self._stance_phase_sample_thd_lower = 200
            self._stance_phase_sample_thd_higher = 400
            self._impact_peak_sample_num_lower = 20
            self._impact_peak_sample_num_higher = 80
            self._20_80_sample_len_lower = 5
            self._20_80_sample_len_higher = 40
            gait_data_100_df = pd.read_csv(fre_100_path + trial_name + '.csv', index_col=False)
            grf_1000_df = pd.read_csv(fre_1000_path + trial_name + '.csv', index_col=False)
            trial_param_df_100 = self.init_trial_params(gait_data_100_df, grf_1000_df, HAISHENG_SENSOR_SAMPLE_RATE)
            pass

    def init_trial_params(self, gait_data_df, grf_1000_df, sensor_sampling_rate):
        # get the corresponding plate data period
        marker_frame = gait_data_df['marker_frame']
        start_vicon, end_vicon = min(marker_frame), max(marker_frame)
        vicon_force_ratio = int(PLATE_SAMPLE_RATE / MOCAP_SAMPLE_RATE)
        start_row_grf = int((start_vicon-1) * vicon_force_ratio)
        end_row_grf = int(end_vicon * vicon_force_ratio) - 1
        plate_data_1000 = grf_1000_df.loc[start_row_grf:end_row_grf].reset_index(drop=True)

        l_strikes, r_strikes, l_offs, r_offs = self.get_strike_off(gait_data_df)
        # get strike off of 1000 Hz data
        l_strikes_1000, r_strikes_1000, l_offs_1000, r_offs_1000 = self.get_strike_off_1000(
            gait_data_df, plate_data_1000, sensor_sampling_rate)
        strike_index_all = self.get_strike_index_all(gait_data_df)
        FPA_all = self.get_FPA_all(gait_data_df)    # FPA of all the samples
        param_data = np.column_stack([l_strikes, r_strikes, l_offs, r_offs, strike_index_all, FPA_all])
        param_data_df = pd.DataFrame(param_data)
        param_data_df.columns = ['l_strikes', 'r_strikes', 'l_offs', 'r_offs', 'l_strike_index', 'r_strike_index',
                                 'l_FPA', 'r_FPA']
        param_data_df.insert(0, 'marker_frame', gait_data_df['marker_frame'])
        # get loading rate
        l_steps_1000 = self.get_legal_steps(l_strikes_1000, l_offs_1000, plate_data_1000, check_steps=False)
        l_LR = self.get_loading_rate(plate_data_1000, l_steps_1000)
        self.insert_LR_to_param_data(param_data_df, l_LR, 'l_LR')
        r_steps_1000 = self.get_legal_steps(r_strikes_1000, r_offs_1000, plate_data_1000, check_steps=False)
        r_LR = self.get_loading_rate(plate_data_1000, r_steps_1000)
        self.insert_LR_to_param_data(param_data_df, r_LR, 'r_LR')
        return param_data_df

    @staticmethod
    def check_loading_rate_all(l_loading_rate):
        plt.figure()
        plt.plot(l_loading_rate)

    def get_strike_off(self, gait_data_df, threshold=50):
        force = gait_data_df[['f_1_x', 'f_1_y', 'f_1_z']].values
        force_norm = norm(force, axis=1)
        strikes = self.get_raw_strikes(force_norm, threshold)
        offs = self.get_raw_offs(force_norm, threshold)
        self.check_strikes_offs(gait_data_df, strikes, offs)

        # distribute strikes offs to left and right foot
        data_len = len(strikes)
        l_heel_y = gait_data_df['LFCC_y']
        r_heel_y = gait_data_df['RFCC_y']
        l_strikes, r_strikes = np.zeros(data_len), np.zeros(data_len)
        l_offs, r_offs = np.zeros(data_len), np.zeros(data_len)
        for i_sample in range(data_len):
            if strikes[i_sample] == 1:
                if l_heel_y[i_sample] > r_heel_y[i_sample]:
                    l_strikes[i_sample] = 1
                else:
                    r_strikes[i_sample] = 1
            if offs[i_sample] == 1:
                if l_heel_y[i_sample] < r_heel_y[i_sample]:
                    l_offs[i_sample] = 1
                else:
                    r_offs[i_sample] = 1
        return l_strikes, r_strikes, l_offs, r_offs

    def get_raw_strikes(self, force_norm, threshold, comparison_len=2):
        data_len = force_norm.shape[0]
        strikes = np.zeros(data_len, dtype=np.int8)
        for i_point in range(comparison_len - 1, data_len - comparison_len):
            if force_norm[i_point - 2] > threshold:
                continue
            if force_norm[i_point - 1] > threshold:
                continue
            if force_norm[i_point] < threshold:
                continue
            if force_norm[i_point + 1] < threshold:
                continue
            if force_norm[i_point + 2] < threshold:
                continue
            strikes[i_point - 1] = 1
        return strikes

    def get_strike_off_1000(self, gait_data_df, plate_data_1000, sensor_sampling_rate, threshold=50):
        force = plate_data_1000[['f_1_x', 'f_1_y', 'f_1_z']].values
        force_norm = norm(force, axis=1)
        strikes = self.get_raw_strikes(force_norm, threshold)
        offs = self.get_raw_offs(force_norm, threshold)
        self.check_strikes_offs(gait_data_df, strikes, offs)

        # distribute strikes offs to left and right foot
        data_len = len(strikes)
        ratio = sensor_sampling_rate / PLATE_SAMPLE_RATE
        l_strikes, r_strikes = np.zeros(data_len), np.zeros(data_len)
        l_offs, r_offs = np.zeros(data_len), np.zeros(data_len)
        for i_sample in range(data_len):
            if strikes[i_sample] == 1:
                l_heel_y = gait_data_df.loc[int(i_sample * ratio), 'LFCC_y']
                r_heel_y = gait_data_df.loc[int(i_sample * ratio), 'RFCC_y']
                if l_heel_y > r_heel_y:
                    l_strikes[i_sample] = 1
                else:
                    r_strikes[i_sample] = 1
            if offs[i_sample] == 1:
                l_heel_y = gait_data_df.loc[int(i_sample * ratio), 'LFCC_y']
                r_heel_y = gait_data_df.loc[int(i_sample * ratio), 'RFCC_y']
                if l_heel_y < r_heel_y:
                    l_offs[i_sample] = 1
                else:
                    r_offs[i_sample] = 1
        return l_strikes, r_strikes, l_offs, r_offs

    def get_raw_offs(self, force_norm, threshold, comparison_len=2):
        data_len = force_norm.shape[0]
        offs = np.zeros(data_len, dtype=np.int8)
        for i_point in range(comparison_len - 1, data_len - comparison_len):
            if force_norm[i_point - 2] < threshold:
                continue
            if force_norm[i_point - 1] < threshold:
                continue
            if force_norm[i_point] > threshold:
                continue
            if force_norm[i_point + 1] > threshold:
                continue
            if force_norm[i_point + 2] > threshold:
                continue
            offs[i_point] = 1
        return offs

    def check_strikes_offs(self, gait_data_df, strikes, offs):
        strike_indexes = np.where(strikes == 1)[0]
        off_indexes = np.where(offs == 1)[0]
        data_len = min(strike_indexes.shape[0], off_indexes.shape[0])

        # check strike off by checking if each strike is followed by a off
        diffs = np.array(strike_indexes[:data_len]) - np.array(off_indexes[:data_len])
        strike_ahead_num, off_ahead_num = 0, 0
        for i_diff in range(data_len):
            if diffs[i_diff] > 0:
                off_ahead_num += 1
            else:
                strike_ahead_num += 1
        if min(off_ahead_num, strike_ahead_num) > 0:
            f_1_z = gait_data_df['f_1_z'].values
            plt.plot(f_1_z)
            plt.plot(strike_indexes, f_1_z[strike_indexes], 'r*')
            plt.plot(off_indexes,  f_1_z[strike_indexes], 'g*')
            plt.show()
            raise ValueError('For trial {trial_name}, {wrong_num} strike off detection result are wrong.'.format(
                trial_name=self._current_trial, wrong_num=min(off_ahead_num, strike_ahead_num)))

    def plot_strikes_offs(self, strikes, offs, check_len=5000):
        l_force = self._gait_data.as_matrix(columns=['f_1_x', 'f_1_y', 'f_1_z'])
        l_force_norm = l_force[0:check_len, 0:2]
        plt.figure()
        plt.plot(norm(l_force_norm, axis=1))

        strike_locs = np.where(strikes[:, 0] == 1)
        off_locs = np.where(offs[:, 0] == 1)
        plt.plot(strike_locs, 1.7, '.', color='red')
        plt.plot(off_locs, 1.7, '.', color='yellow')

        # for i in range(0, check_len):
        #     if strikes[i, 0] == 1:
        #         plt.plot(i, 1.7, '.', color='red')
        #     if offs[i, 0] == 1:
        #         plt.plot(i, 1.7, '.', color='yellow')
        # plt.legend()
        plt.title('heel strike & toe off')

    def get_trunk_swag(self):
        C7 = self._gait_data.as_matrix(columns=['C7_x', 'C7_y', 'C7_z'])
        l_PSIS = self._gait_data.as_matrix(columns=['l_PSIS_x', 'l_PSIS_y', 'l_PSIS_z'])
        r_PSIS = self._gait_data.as_matrix(columns=['r_PSIS_x', 'r_PSIS_y', 'r_PSIS_z'])
        middle_PSIS = (l_PSIS + r_PSIS) / 2
        vertical_vector = C7 - middle_PSIS
        return - 180 / np.pi * np.arctan(vertical_vector[:, 0] / vertical_vector[:, 2])

    @staticmethod
    def check_trunk_swag(data_trunk_swag):
        plt.figure()
        plt.plot(data_trunk_swag)
        plt.title('trunk swag')

    def get_strike_index_all(self, gait_data_df):
        l_foot_data = gait_data_df[['LFM2_y', 'LFCC_y', 'c_1_y']].values
        l_foot_length = l_foot_data[:, 0] - l_foot_data[:, 1]
        l_cop_length = l_foot_data[:, 2] - l_foot_data[:, 1]
        l_index = l_cop_length / l_foot_length
        l_index[l_index < 0] = 0
        l_index[l_index > 1] = 1

        r_foot_data = gait_data_df[['RFM2_y', 'RFCC_y', 'c_1_y']].values
        r_foot_length = r_foot_data[:, 0] - r_foot_data[:, 1]
        r_cop_length = r_foot_data[:, 2] - r_foot_data[:, 1]
        r_index = r_cop_length / r_foot_length
        r_index[r_index < 0] = 0
        r_index[r_index > 1] = 1

        return np.column_stack([l_index, r_index])

    def get_foot_strike_angle(self):
        static_data_path = PROCESSED_DATA_PATH + self._sub_name + '\\' + FILE_NAMES[0] + '.csv'
        static_param_processor = ParamProcessor(static_data_path, 0, 0)

        l_toe_static = static_param_processor._gait_data.as_matrix(columns=['l_toe_x', 'l_toe_y', 'l_toe_z'])
        l_heel_static = static_param_processor._gait_data.as_matrix(columns=['l_heel_x', 'l_heel_y', 'l_heel_z'])
        l_foot_diff = l_toe_static - l_heel_static
        l_foot_len = np.mean(norm(l_foot_diff, axis=1))
        l_toe_z_static = np.mean(l_toe_static[:, 2])
        l_heel_z_static = np.mean(l_heel_static[:, 2])
        l_toe_z = self._gait_data.as_matrix(columns=['l_toe_z'])
        l_heel_z = self._gait_data.as_matrix(columns=['l_heel_z'])
        z_diff = (l_toe_z - l_toe_z_static) - (l_heel_z - l_heel_z_static)
        l_foot_strike_angle = np.rad2deg(np.arcsin(z_diff / l_foot_len))

        r_toe_static = static_param_processor._gait_data.as_matrix(columns=['r_toe_x', 'r_toe_y', 'r_toe_z'])
        r_heel_static = static_param_processor._gait_data.as_matrix(columns=['r_heel_x', 'r_heel_y', 'r_heel_z'])
        r_foot_diff = r_toe_static - r_heel_static
        r_foot_len = np.mean(norm(r_foot_diff, axis=1))
        r_toe_z_static = np.mean(r_toe_static[:, 2])
        r_heel_z_static = np.mean(r_heel_static[:, 2])
        r_toe_z = self._gait_data.as_matrix(columns=['r_toe_z'])
        r_heel_z = self._gait_data.as_matrix(columns=['r_heel_z'])
        z_diff = (r_toe_z - r_toe_z_static) - (r_heel_z - r_heel_z_static)
        r_foot_strike_angle = np.rad2deg(np.arcsin(z_diff / r_foot_len))

        return np.column_stack([l_foot_strike_angle, r_foot_strike_angle])

    # this program use 1000Hz force plate data
    def get_loading_rate(self, plate_data, steps):
        # 20% to 80% from strike to impact peak
        loading_rates = []
        grf_z = plate_data['f_1_z'].values
        for step in steps:
            grf_z_step = grf_z[step[0]:step[1]]
            peaks, _ = find_peaks(-grf_z_step, height=200, prominence=150)
            try:    # find legal peaks
                if len(peaks) == 1:     # case 0, no impact peak, only one max peak
                    impact_peak_sample_num = 0.13 * (step[1] - step[0])
                elif len(peaks) == 2:     # case 1, impact peak exists
                    impact_peak_sample_num = peaks[0]
                else:
                    raise ValueError('Wrong peak number, please check the plot.')
                if impact_peak_sample_num < self._impact_peak_sample_num_lower or\
                        impact_peak_sample_num > self._impact_peak_sample_num_higher:
                    raise ValueError('Wrong impact peak, please check the plot.')
            except ValueError as value_error:
                print(value_error.args[0])
                plt.plot(grf_z_step)
                plt.plot(peaks, grf_z_step[peaks], 'r*')
                plt.show()
            peak_index = int(impact_peak_sample_num)
            impact_peak_force = grf_z_step[peak_index]
            force_start = 0.2 * impact_peak_force
            start_index = np.abs(grf_z_step[:peak_index] - force_start).argmin()
            force_end = 0.8 * impact_peak_force
            end_index = np.abs(grf_z_step[:peak_index] - force_end).argmin()
            try:
                if end_index - start_index < self._20_80_sample_len_lower or \
                        end_index - start_index > self._20_80_sample_len_higher:
                    raise ValueError('Wrong 20% - 80% sample num, found {num} in total, please check the plot'.
                                     format(num=end_index - start_index))
            except ValueError as value_error:
                print(value_error.args[0])
                plt.plot(grf_z_step)
                plt.plot([start_index, end_index], [grf_z_step[start_index], grf_z_step[end_index]], 'r-')
                plt.show()
            loading_rate = (grf_z_step[end_index] - grf_z_step[start_index]) / (end_index - start_index)
            if LOADING_RATE_NORMALIZATION:
                loading_rate = - loading_rate / self.__weight
            marker_frame = plate_data.loc[int((step[0] + step[1]) / 2), 'marker_frame']
            loading_rates.append([loading_rate, marker_frame])
        return loading_rates
        #     plt.plot(grf_z_step)
        #     plt.plot([int(impact_peak_sample_num), int(impact_peak_sample_num)],
        #              [0, grf_z_step[int(impact_peak_sample_num)]], 'd--')
        # plt.show()

    def insert_params_to_param_data(self, param_data_df, params, param_names):
        if len(params) != len(param_names):
            raise RuntimeError('The param number and param name number doesn\'t match')
        for param, param_name in params, param_names:
            param_data_df.insert(len(param_data_df.columns), param_name, param)

    def insert_LR_to_param_data(self, gait_data_df, insert_list, column_name):
        data_len = gait_data_df.shape[0]
        insert_data = np.zeros([data_len])
        for item in insert_list:
            row_index = gait_data_df.index[gait_data_df['marker_frame'] == item[1]]
            if len(row_index) == 0:
                row_index = gait_data_df.index[gait_data_df['marker_frame'] == item[1] + 1]
            insert_data[row_index[0]] = item[0]
        gait_data_df.insert(len(gait_data_df.columns), column_name, insert_data)

    def get_legal_steps(self, strikes, offs, plate_data=None, check_steps=False):
        """
        Sometimes subjects have their both feet on the ground so it is necessary to do step check.
        :param strikes: 
        :param offs: 
        :param plate_data: 
        :param check_steps: 
        :return: 
        """
        strike_tuple = np.where(strikes == 1)[0]
        off_tuple = np.where(offs == 1)[0]
        off_tuple = off_tuple[off_tuple > strike_tuple[0]]
        steps = []
        for i_step in range(min(len(strike_tuple), len(off_tuple))):
            stance_start = strike_tuple[i_step]
            stance_end = off_tuple[i_step]
            # pop out illegal steps
            if self._stance_phase_sample_thd_lower > stance_end - stance_start:
                off_tuple = np.delete(off_tuple, i_step)
            if stance_end - stance_start > self._stance_phase_sample_thd_higher:
                strike_tuple = np.delete(strike_tuple, i_step)
            steps.append([strike_tuple[i_step], off_tuple[i_step]])
        if check_steps:
            grf_z = plate_data['f_1_z'].values
            for step in steps:
                plt.plot(grf_z[step[0]:step[1]])
            plt.show()
        return steps

    def get_loading_rate_all_100Hz(self):
        data_len = self._data_len
        #  - 1e-10 is used to fix a strange problem
        LR_step = np.arange(0, data_len / self._sampling_rate - 1e-10, 1 / self._sampling_rate)

        l_force_z = self._gait_data.as_matrix(columns=['f_1_z'])
        tck, LR_step = interpo.splprep(l_force_z.T, u=LR_step, s=0)
        l_loading_rate_all_raw = interpo.splev(LR_step, tck, der=1)  # der=1 means take the first derivation
        l_loading_rate_all = l_loading_rate_all_raw[0]

        r_force_z = self._gait_data.as_matrix(columns=['f_2_z'])
        tck, LR_step = interpo.splprep(r_force_z.T, u=LR_step, s=0)
        r_loading_rate_all_raw = interpo.splev(LR_step, tck, der=1)  # der=1 means take the first derivation
        r_loading_rate_all = r_loading_rate_all_raw[0]

        return np.column_stack([l_loading_rate_all, r_loading_rate_all])

    # def get_strike_off_from_imu(self, i_trial):
    #     my_detector = Detectors(self._sub_name, 'l')
    #     if i_trial < 4:
    #         l_strike_esti, l_off_esti = my_detector.walking_tape_strike_off(TRIAL_NAMES[i_trial])
    #     else:
    #         l_strike_esti = my_detector.jogging_tape_strike(TRIAL_NAMES[i_trial])
    #         l_off_esti = [0]
    #     l_strikes = np.zeros([self._data_len, 1])
    #     l_strikes[l_strike_esti] = 1
    #     l_offs = np.zeros([self._data_len, 1])
    #     l_offs[l_off_esti] = 1
    #     return np.column_stack([l_strikes, l_offs])

    @staticmethod
    def compare_100Hz_1000Hz(l_1000, r_1000, l_100, r_100):
        plt.figure()
        plt.plot(l_1000)
        plt.plot(l_100)
        plt.figure()
        plt.plot(r_1000)
        plt.plot(r_100)
        plt.show()

    def get_step_width(self):
        l_ankle_l = self._gait_data.as_matrix(columns=['l_ankle_l_x', 'l_ankle_l_y', 'l_ankle_l_z'])
        l_ankle_r = self._gait_data.as_matrix(columns=['l_ankle_r_x', 'l_ankle_r_y', 'l_ankle_r_z'])
        r_ankle_l = self._gait_data.as_matrix(columns=['r_ankle_l_x', 'r_ankle_l_y', 'r_ankle_l_z'])
        r_ankle_r = self._gait_data.as_matrix(columns=['r_ankle_r_x', 'r_ankle_r_y', 'r_ankle_r_z'])
        data_len = l_ankle_l.shape[0]
        step_width = np.zeros(data_len)
        heel_strikes = self.get_heel_strike_event()
        # set the right feet as dominate feet
        new_step = False
        for i_point in range(0, data_len):
            # check left foot
            if heel_strikes[i_point, 0] == 1:
                ankle_l = (l_ankle_l[i_point, 0] + l_ankle_r[i_point, 0]) / 2
                new_step = True
            # check right foot
            if heel_strikes[i_point, 1] == 1:
                if new_step:
                    ankle_r = (r_ankle_l[i_point, 0] + r_ankle_r[i_point, 0]) / 2
                    step_width[i_point] = ankle_r - ankle_l
                    new_step = False

        return step_width

    def check_step_width(self, step_widths, check_len=5000):
        forces = self.get_force()
        plt.figure()
        plt.plot(norm(forces[0:check_len, 3:5], axis=1))
        for i in range(0, check_len):
            if step_widths[i] != 0:
                plt.plot(i, 1.7, '.', color='red')
        plt.legend()
        plt.title('step width')

    # FPA of all the samples
    def get_FPA_all(self, gait_data_df):
        l_toe = gait_data_df[['LFM2_x', 'LFM2_y', 'LFM2_z']].values
        l_heel = gait_data_df[['LFCC_x', 'LFCC_y', 'LFCC_z']].values
        data_len = l_toe.shape[0]
        left_FPAs = np.zeros(data_len)
        for i_point in range(0, data_len):
            forward_vector = l_toe[i_point, :] - l_heel[i_point, :]
            left_FPAs[i_point] = - 180 / np.pi * np.arctan(forward_vector[0] / forward_vector[1])

        r_toe = gait_data_df[['RFM2_x', 'RFM2_y', 'RFM2_z']].values
        r_heel = gait_data_df[['RFCC_x', 'RFCC_y', 'RFCC_z']].values
        right_FPAs = np.zeros(data_len)
        for i_point in range(0, data_len):
            forward_vector = r_toe[i_point, :] - r_heel[i_point, :]
            right_FPAs[i_point] = 180 / np.pi * np.arctan(forward_vector[0] / forward_vector[1])

        return np.column_stack([left_FPAs, right_FPAs])

    def get_steps(self):
        heel_strikes = self.get_heel_strike_event()
        strike_index = np.where(heel_strikes == 1)[0]
        step_num = len(strike_index) - 1
        return strike_index, step_num

    def check_FPA(self, FPAs, check_len=5000):
        forces = self.get_force()
        plt.figure()
        plt.plot(norm(forces[0:check_len, 0:2], axis=1))
        for i in range(0, check_len):
            if FPAs[i, 0] != 0:
                plt.plot(i, 1.7, '.', color='red')
        plt.legend()
        plt.title('FPA')

    @staticmethod
    def check_FPA_all(FPAs):
        plt.figure()
        plt.plot(FPAs)
        plt.legend()
        plt.title('FPA')

    # be careful about absent values
    def get_pelvis_angle(self):
        l_PSIS = self._gait_data.as_matrix(columns=['l_PSIS_x', 'l_PSIS_y', 'l_PSIS_z'])
        r_PSIS = self._gait_data.as_matrix(columns=['r_PSIS_x', 'r_PSIS_y', 'r_PSIS_z'])
        l_ASIS = self._gait_data.as_matrix(columns=['l_ASIS_x', 'l_ASIS_y', 'l_ASIS_z'])
        r_ASIS = self._gait_data.as_matrix(columns=['r_ASIS_x', 'r_ASIS_y', 'r_ASIS_z'])

        # # posterior side offset
        # l_PSIS[:, 2] = l_PSIS[:, 2] - 35
        # r_PSIS[:, 2] = r_PSIS[:, 2] - 35

        x_vector = r_ASIS - l_ASIS
        y_vector = (l_ASIS + r_ASIS) / 2 - (l_PSIS + r_PSIS) / 2
        x_vector_norm = x_vector / norm(x_vector, axis=1)[:, None]
        y_vector_norm = y_vector / norm(y_vector, axis=1)[:, None]
        z_vector_norm = np.cross(x_vector_norm, y_vector_norm)
        alpha = 180 / np.pi * np.arctan2(-z_vector_norm[:, 1], z_vector_norm[:, 2])
        beta = 180 / np.pi * np.arcsin(z_vector_norm[:, 0])
        gamma = 180 / np.pi * np.arctan(-y_vector_norm[:, 0], x_vector_norm[:, 0])
        return np.column_stack([alpha, beta, gamma])

    @staticmethod
    def check_pelvis_angle(pelvis_angles):
        plt.figure()
        plt.plot(pelvis_angles)
        plt.title('pelvis angle')

    def get_knee_flexion_angle(self):
        l_knee_l = self._gait_data.as_matrix(columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])
        l_ankle_l = self._gait_data.as_matrix(columns=['l_ankle_l_x', 'l_ankle_l_y', 'l_ankle_l_z'])
        l_hip = self._gait_data.as_matrix(columns=['l_hip_x', 'l_hip_y', 'l_hip_z'])
        shank_vector = l_ankle_l[:, 1:] - l_knee_l[:, 1:]
        thigh_vector = l_hip[:, 1:] - l_knee_l[:, 1:]
        l_knee_angles = 180 - self.__law_of_cosines(shank_vector, thigh_vector)

        r_knee_r = self._gait_data.as_matrix(columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])
        r_ankle_r = self._gait_data.as_matrix(columns=['r_ankle_r_x', 'r_ankle_r_y', 'r_ankle_r_z'])
        r_hip = self._gait_data.as_matrix(columns=['r_hip_x', 'r_hip_y', 'r_hip_z'])
        shank_vector = r_ankle_r[:, 1:] - r_knee_r[:, 1:]
        thigh_vector = r_hip[:, 1:] - r_knee_r[:, 1:]
        r_knee_angles = 180 - self.__law_of_cosines(shank_vector, thigh_vector)

        return np.column_stack([l_knee_angles, r_knee_angles])

    def check_knee_flexion_angle(self, knee_flexion_angle):
        plt.figure()
        plt.plot(knee_flexion_angle)
        plt.title('knee flexion angle')

    def get_ankle_flexion_angle(self):
        l_knee_l = self._gait_data.as_matrix(columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])
        l_knee_r = self._gait_data.as_matrix(columns=['l_knee_r_x', 'l_knee_r_y', 'l_knee_r_z'])
        l_ankle_l = self._gait_data.as_matrix(columns=['l_ankle_l_x', 'l_ankle_l_y', 'l_ankle_l_z'])
        l_ankle_r = self._gait_data.as_matrix(columns=['l_ankle_r_x', 'l_ankle_r_y', 'l_ankle_r_z'])
        l_toe_mt2 = self._gait_data.as_matrix(columns=['l_toe_x', 'l_toe_y', 'l_toe_z'])
        l_cal = self._gait_data.as_matrix(columns=['l_cal_x', 'l_cal_y', 'l_cal_z'])
        l_knee_center = (l_knee_l[:, 1:] + l_knee_r[:, 1:]) / 2
        l_ankle_center = (l_ankle_l[:, 1:] + l_ankle_r[:, 1:]) / 2
        l_shank_vector = l_ankle_center - l_knee_center
        l_foot_vector = l_toe_mt2[:, 1:] - l_cal[:, 1:]
        l_ankle_angles = self.__law_of_cosines(l_shank_vector, l_foot_vector) - 90

        r_knee_l = self._gait_data.as_matrix(columns=['r_knee_l_x', 'r_knee_l_y', 'r_knee_l_z'])
        r_knee_r = self._gait_data.as_matrix(columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])
        r_ankle_l = self._gait_data.as_matrix(columns=['r_ankle_l_x', 'r_ankle_l_y', 'r_ankle_l_z'])
        r_ankle_r = self._gait_data.as_matrix(columns=['r_ankle_r_x', 'r_ankle_r_y', 'r_ankle_r_z'])
        r_toe_mt2 = self._gait_data.as_matrix(columns=['r_toe_x', 'r_toe_y', 'r_toe_z'])
        r_cal = self._gait_data.as_matrix(columns=['r_cal_x', 'r_cal_y', 'r_cal_z'])
        r_knee_center = (r_knee_l[:, 1:] + r_knee_r[:, 1:]) / 2
        r_ankle_center = (r_ankle_l[:, 1:] + r_ankle_r[:, 1:]) / 2
        r_shank_vector = r_ankle_center - r_knee_center
        r_foot_vector = r_toe_mt2[:, 1:] - r_cal[:, 1:]
        r_ankle_angles = self.__law_of_cosines(r_shank_vector, r_foot_vector) - 90

        return np.column_stack([l_ankle_angles, r_ankle_angles])

    def check_ankle_flexion_angle(self, ankle_flexion_angle):
        plt.figure()
        plt.plot(ankle_flexion_angle)
        plt.title('ankle flexion angle')

    @staticmethod
    def __law_of_cosines(vector1, vector2):
        vector3 = vector1 - vector2
        num = inner1d(vector1, vector1) + \
              inner1d(vector2, vector2) - inner1d(vector3, vector3)
        den = 2 * np.sqrt(inner1d(vector1, vector1)) * np.sqrt(inner1d(vector2, vector2))
        return 180 / np.pi * np.arccos(num / den)

    def get_hip_flexion_angle(self):
        l_PSIS = self._gait_data.as_matrix(columns=['l_PSIS_x', 'l_PSIS_y', 'l_PSIS_z'])
        r_PSIS = self._gait_data.as_matrix(columns=['r_PSIS_x', 'r_PSIS_y', 'r_PSIS_z'])
        C7 = self._gait_data.as_matrix(columns=['C7_x', 'C7_y', 'C7_z'])
        middle_PSIS = (l_PSIS[:, 1:] + r_PSIS[:, 1:]) / 2
        vertical_vector = C7[:, 1:] - middle_PSIS

        l_knee_l = self._gait_data.as_matrix(columns=['l_knee_l_x', 'l_knee_l_y', 'l_knee_l_z'])
        l_hip = self._gait_data.as_matrix(columns=['l_hip_x', 'l_hip_y', 'l_hip_z'])
        thigh_vector = l_hip[:, 1:] - l_knee_l[:, 1:]
        l_hip_angle = 180 - self.__law_of_cosines(vertical_vector, thigh_vector)

        r_knee_r = self._gait_data.as_matrix(columns=['r_knee_r_x', 'r_knee_r_y', 'r_knee_r_z'])
        r_hip = self._gait_data.as_matrix(columns=['r_hip_x', 'r_hip_y', 'r_hip_z'])
        thigh_vector = r_hip[:, 1:] - r_knee_r[:, 1:]
        r_hip_angle = 180 - self.__law_of_cosines(vertical_vector, thigh_vector)

        return np.column_stack([l_hip_angle, r_hip_angle])

    def check_hip_flexion_angle(self, hip_flexion_angle):
        plt.figure()
        plt.plot(hip_flexion_angle)
        plt.title('hip flexion angle')

    # this code can be used for ground walking
    def get_ground_FPAs(self):
        l_toe = self._gait_data.as_matrix(columns=['l_toe_x', 'l_toe_y', 'l_toe_z'])
        l_heel = self._gait_data.as_matrix(columns=['l_heel_x', 'l_heel_y', 'l_heel_z'])
        data_len = l_toe.shape[0]
        left_FPAs = np.zeros(data_len)
        strike_index, step_num = self.get_steps()
        for i_step in range(step_num):
            heading_vector = l_heel[i_step+1, :] - l_heel[i_step, :]
            heading_vector = heading_vector[0:2]
            foot_vector = l_toe[i_step+1, :] - l_heel[i_step+1, :]
            foot_vector = foot_vector[0:2]
            rest_vector = heading_vector - foot_vector
            numerator = (norm(heading_vector)**2 + norm(foot_vector)**2 - norm(rest_vector)**2)
            denominator = (2 * np.dot(heading_vector, foot_vector))
            FPA = - 180 / np.pi * np.arccos(numerator / denominator)
            left_FPAs[strike_index[i_step+1]] = FPA













