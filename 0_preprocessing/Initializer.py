import pandas as pd
import numpy as np
from const import RAW_DATA_PATH, TRIAL_NAMES, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE, \
    XSENS_SENSOR_LOACTIONS, XSENS_FILE_NAME_DIC, STATIC_STANDING_PERIOD, DATA_COLUMNS_IMU, SEGMENT_MARKERS
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os
import xlrd
from ViconReader import ViconReader
from HaishengSensorReader import HaishengSensorReader
from XsensReader import XsensReader
from GyrSimulator import GyrSimulator


class SubjectDataInitializer:
    def __init__(self, processed_data_path, subject_folder, trials_100hz, readme_xls, check_sync=False,
                 check_running_period=False, initialize_100Hz=False, initialize_200Hz=False, initialize_1000Hz=False):
        print('Subject: ' + subject_folder)
        self.__processed_data_path = processed_data_path
        self._subject_folder = subject_folder
        self.__readme_xls = readme_xls

        fre_100_path, fre_200_path, fre_1000_path = SubjectDataInitializer._initialize_path(
            processed_data_path, subject_folder)

        # initialize 100 Hz data
        if initialize_100Hz:
            HaishengSensorReader.rename_haisheng_sensor_files(RAW_DATA_PATH + subject_folder + '\\haisheng',
                                                              readme_xls)
            for trial_name in trials_100hz:
                print('Initializing {trial_name} trial, vicon and Haisheng, 100 Hz...'.format(trial_name=trial_name))
                vicon_all_df, l_foot_marker_df, r_foot_marker_df, start_vicon, end_vicon = \
                    self.initialize_vicon_resampled(trial_name, HAISHENG_SENSOR_SAMPLE_RATE, check_running_period)
                haisheng_df = self.initialize_haisheng_sensor(
                    trial_name, r_foot_marker_df, start_vicon, end_vicon)
                data_all_df = pd.concat([vicon_all_df, haisheng_df], axis=1)
                SubjectDataInitializer.__save_data(fre_100_path, trial_name, data_all_df)
                if check_sync:
                    self.check_sync(trial_name, vicon_all_df, haisheng_df, 'r_foot', HAISHENG_SENSOR_SAMPLE_RATE)
            plt.show()
            print('100 Hz data done. Please check plots.')

        # initialize 200 Hz data
        if initialize_200Hz:
            for trial_name in TRIAL_NAMES:
                print('Initializing {trial_name} trial, vicon and xsens, 200 Hz...'.format(trial_name=trial_name))
                vicon_all_df, l_foot_marker_df, r_foot_marker_df, start_vicon, end_vicon = \
                    self.initialize_vicon(trial_name, check_running_period=check_running_period)
                xsens_all_df = self.initialize_xsens(trial_name, l_foot_marker_df, start_vicon, end_vicon)
                data_all_df = pd.concat([vicon_all_df, xsens_all_df], axis=1)
                SubjectDataInitializer.__save_data(fre_200_path, trial_name, data_all_df)
                if check_sync:
                    self.check_sync(trial_name, vicon_all_df, xsens_all_df, 'l_foot', MOCAP_SAMPLE_RATE)
            plt.show()
            print('200 Hz data done. Please check plots.')

        # initialzed 1000 Hz GRFz data, all the data was saved
        if initialize_1000Hz:
            for trial_name in TRIAL_NAMES:
                print('Initializing {trial_name} trial, z-axis of GRF, 1000 Hz...'.format(trial_name=trial_name))
                file_path_vicon = '{path}{sub_folder}\\{sensor}\\{file_name}.csv'.format(
                    path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='vicon', file_name=trial_name)
                vicon_reader = ViconReader(file_path_vicon)
                grf_1000Hz = vicon_reader.get_plate_processed()
                SubjectDataInitializer.__save_data(fre_1000_path, trial_name, grf_1000Hz)
        plt.show()

    def initialize_haisheng_sensor(self, trial_name, marker_df, start_vicon, end_vicon):
        sensor_all_df = pd.DataFrame()
        for location in ['r_foot']:
            file_path_haisheng = '{path}{sub_folder}\\{sensor}\\{sensor_loc}_renamed\\{trial_name}.csv'.format(
                path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='haisheng', sensor_loc=location,
                trial_name=trial_name)
            haisheng_sensor_reader = HaishengSensorReader(file_path_haisheng)
            sensor_gyr_norm = haisheng_sensor_reader.get_normalized_gyr()

            # get gyr norm from simulation
            my_nike_gyr_simulator = GyrSimulator(self._subject_folder, location)
            gyr_vicon = my_nike_gyr_simulator.get_gyr(trial_name, marker_df, sampling_rate=HAISHENG_SENSOR_SAMPLE_RATE)
            gyr_norm_vicon = norm(gyr_vicon, axis=1)

            # in vicon data, the first 20 samples can be very noisy
            vicon_delay = GyrSimulator.sync_vicon_sensor(trial_name, 'r_foot', gyr_norm_vicon[20:],
                                                         sensor_gyr_norm[20:], check=False)
            start_haisheng, end_haisheng = start_vicon + vicon_delay, end_vicon + vicon_delay
            sensor_df = haisheng_sensor_reader.data_processed_df.copy().loc[start_haisheng:end_haisheng]
            sensor_df = sensor_df.drop(['sample'], axis=1).reset_index(drop=True)
            current_xsens_col_names = [location + '_' + channel for channel in DATA_COLUMNS_IMU]
            sensor_df.columns = current_xsens_col_names
            sensor_all_df = pd.concat([sensor_df, sensor_all_df], axis=1)

        return sensor_all_df

    def initialize_xsens(self, trial_name, l_foot_marker_df, start_vicon, end_vicon, check=False):
        # get gyr norm from left foot xsens sensor
        file_path_xsens = '{path}{sub_folder}\\{sensor}\\{trial_folder}\\'.format(
            path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='xsens', trial_folder=trial_name)
        l_foot_xsens_reader = XsensReader(file_path_xsens + XSENS_FILE_NAME_DIC['l_foot'], self._subject_folder,
                                          'l_foot', trial_name)
        sensor_gyr_norm = l_foot_xsens_reader.get_normalized_gyr()
        # get gyr norm from simulation
        my_nike_gyr_simulator = GyrSimulator(self._subject_folder, 'l_foot')
        gyr_vicon = my_nike_gyr_simulator.get_gyr(trial_name, l_foot_marker_df, sampling_rate=MOCAP_SAMPLE_RATE)
        gyr_norm_vicon = norm(gyr_vicon, axis=1)
        # in vicon data, the first 20 samples can be very noisy
        vicon_delay = GyrSimulator.sync_vicon_sensor(trial_name, 'l_foot', gyr_norm_vicon[20:], sensor_gyr_norm[20:],
                                                     check)
        start_xsens, end_xsens = start_vicon + vicon_delay, end_vicon + vicon_delay

        xsens_all_df = pd.DataFrame()
        for xsens_location in XSENS_SENSOR_LOACTIONS:
            current_xsens_col_names = [xsens_location + '_' + channel for channel in DATA_COLUMNS_IMU]
            xsens_reader = XsensReader(file_path_xsens + XSENS_FILE_NAME_DIC[xsens_location], self._subject_folder,
                                       xsens_location, trial_name)
            current_xsens_all_df = xsens_reader.data_processed_df
            current_xsens_df = current_xsens_all_df.copy().loc[start_xsens:end_xsens].reset_index(drop=True)
            current_xsens_df.columns = current_xsens_col_names
            xsens_all_df = pd.concat([xsens_all_df, current_xsens_df], axis=1)
        return xsens_all_df

    def initialize_vicon(self, trial_name, check_running_period=False):
        file_path_vicon = '{path}{sub_folder}\\{sensor}\\{file_name}.csv'.format(
            path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='vicon', file_name=trial_name)
        vicon_reader = ViconReader(file_path_vicon)
        # self.plot_trajectory(vicon_reader.marker_data_processed_df['LFCC_y'])

        if 'static' in trial_name:
            # 4 second preparation time
            start_vicon, end_vicon = 5 * MOCAP_SAMPLE_RATE, STATIC_STANDING_PERIOD * MOCAP_SAMPLE_RATE
        elif 'SI' in trial_name:
            start_vicon, end_vicon = self.__find_three_pattern_period(
                vicon_reader.get_plate_data_resampled(), self.__readme_xls, trial_name, MOCAP_SAMPLE_RATE)
        else:  # baseline or strike
            start_vicon, end_vicon = self.__find_running_period(vicon_reader.marker_data_processed_df['LFCC_y'],
                                                                running_thd=300)

        # sometimes the subject start walking on the wrong side of the treadmill so overwrite start_vicon was necessary
        # sometimes the xsens lost connection so overwrite end_vicon was necessary
        # overwrite the start_vicon, end_vicon if they are contained in the readme xls
        readme_sheet = xlrd.open_workbook(self.__readme_xls).sheet_by_index(0)
        trial_num = TRIAL_NAMES.index(trial_name)
        pattern_start = readme_sheet.row_values(trial_num + 2)[13]
        pattern_end = readme_sheet.row_values(trial_num + 2)[14]
        if pattern_start is not '':
            start_vicon = int(pattern_start)
        if pattern_end is not '':
            end_vicon = int(pattern_end)

        if check_running_period:
            f_1_z_data = vicon_reader.get_plate_data_resampled()['f_1_z']
            plt.figure()
            plt.plot(f_1_z_data)
            plt.plot([start_vicon, start_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'g--')
            plt.plot([end_vicon, end_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'r--')

        vicon_all_df = vicon_reader.get_vicon_all_processed_df()
        vicon_all_df = vicon_all_df.loc[start_vicon:end_vicon].reset_index(drop=True)
        l_foot_marker_df = vicon_reader.get_marker_data_processed_segment('l_foot')
        r_foot_marker_df = vicon_reader.get_marker_data_processed_segment('r_foot')
        return vicon_all_df, l_foot_marker_df, r_foot_marker_df, start_vicon, end_vicon

    def initialize_vicon_resampled(self, trial_name, sampling_rate, check_running_period):
        file_path_vicon = '{path}{sub_folder}\\{sensor}\\{file_name}.csv'.format(
            path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='vicon', file_name=trial_name)
        vicon_reader = ViconReader(file_path_vicon)

        if 'static' in trial_name:
            # 4 second preparation time
            start_vicon, end_vicon = 5 * MOCAP_SAMPLE_RATE, STATIC_STANDING_PERIOD * MOCAP_SAMPLE_RATE
        elif 'SI' in trial_name:
            start_vicon, end_vicon = self.__find_three_pattern_period(
                vicon_reader.get_plate_data_resampled(), self.__readme_xls, trial_name, HAISHENG_SENSOR_SAMPLE_RATE)
        else:  # baseline or strike
            start_vicon, end_vicon = self.__find_running_period(vicon_reader.marker_data_processed_df['LFCC_y'],
                                                                running_thd=300)

        # sometimes the subject start walking on the wrong side of the treadmill so overwrite start_vicon was necessary
        # overwrite the end_vicon if they are contained in the readme xls
        readme_sheet = xlrd.open_workbook(self.__readme_xls).sheet_by_index(0)
        trial_num = TRIAL_NAMES.index(trial_name)
        pattern_start = readme_sheet.row_values(trial_num + 2)[13]
        if pattern_start is not '':
            start_vicon = int(pattern_start)

        if check_running_period:
            f_1_z_data = vicon_reader.get_plate_data_resampled()['f_1_z']
            plt.figure()
            plt.plot(f_1_z_data)
            plt.plot([start_vicon, start_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'g--')
            plt.plot([end_vicon, end_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'r--')
            plt.title(trial_name)

        start_vicon = int(start_vicon / (MOCAP_SAMPLE_RATE / sampling_rate))
        end_vicon = int(end_vicon / (MOCAP_SAMPLE_RATE / sampling_rate))
        vicon_all_df = vicon_reader.get_vicon_all_processed_df()
        vicon_all_df = ViconReader.resample_data(vicon_all_df, sampling_rate, MOCAP_SAMPLE_RATE)
        vicon_all_df = vicon_all_df.loc[start_vicon:end_vicon].reset_index(drop=True)
        l_foot_marker_df = vicon_reader.get_marker_data_processed_segment('l_foot')
        l_foot_marker_df = ViconReader.resample_data(l_foot_marker_df, sampling_rate, MOCAP_SAMPLE_RATE)
        r_foot_marker_df = vicon_reader.get_marker_data_processed_segment('r_foot')
        r_foot_marker_df = ViconReader.resample_data(r_foot_marker_df, sampling_rate, MOCAP_SAMPLE_RATE)
        return vicon_all_df, l_foot_marker_df, r_foot_marker_df, start_vicon, end_vicon

    def check_sync(self, trial_name, marker_df, sensor_df, location, sampling_rate=MOCAP_SAMPLE_RATE, check_len=1000):
        """
        If the static trial doesn't match, it is acceptable as long as the value are all lower than 1.
        :param trial_name:
        :param marker_df:
        :param sensor_df:
        :param location:
        :param sampling_rate:
        :param check_len:
        :return:
        """
        segment_marker_names = SEGMENT_MARKERS[location]
        segment_marker_names_xyz = [name + axis for name in segment_marker_names for axis in ['_x', '_y', '_z']]
        marker_df_clip = marker_df[segment_marker_names_xyz].copy().reset_index(drop=True).loc[:check_len]
        if location not in ['r_foot', 'trunk', 'l_foot']:
            raise ValueError('Wrong sensor location')

        gyr_column_names = [location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        sensor_df = sensor_df[gyr_column_names].loc[:check_len]
        # get gyr norm from simulation
        my_nike_gyr_simulator = GyrSimulator(self._subject_folder, location)
        gyr_vicon = my_nike_gyr_simulator.get_gyr(trial_name, marker_df_clip, sampling_rate=sampling_rate)
        gyr_norm_vicon = norm(gyr_vicon, axis=1)
        gyr_norm_sensor = norm(sensor_df, axis=1)
        plt.figure()
        plt.plot(gyr_norm_vicon)
        plt.plot(gyr_norm_sensor)
        plt.title(trial_name)

    @staticmethod
    def plot_trajectory(marker_df):
        # just for testing
        plt.plot(marker_df)
        plt.show()

    @staticmethod
    def __find_three_pattern_period(plate_df, readme_xls, trial_name, sampling_rate, force_thd=200):
        # find the start time when subject stepped on the first force plate
        f_1_z = abs(plate_df['f_1_z'].values)

        # find the end of three patterns from readme
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        trial_num = TRIAL_NAMES.index(trial_name)
        start_vicon = np.argmax(f_1_z > force_thd)
        start_vicon = start_vicon + 2 * sampling_rate

        pattern_ends = readme_sheet.row_values(trial_num + 2)[8:11]
        end_vicon = int(start_vicon + max(pattern_ends))
        return start_vicon, end_vicon

    @staticmethod
    def __find_running_period(marker_df, running_thd, clip_len=200, padding=1200):
        # find running period via marker variance
        marker_mat = marker_df.values
        is_running = np.zeros([int(marker_mat.shape[0] / clip_len)])
        for i_clip in range(len(is_running)):
            data_clip = marker_mat[i_clip * clip_len:(i_clip + 1) * clip_len]
            if np.max(data_clip) - np.min(data_clip) > running_thd:
                is_running[i_clip] = 1

        max_clip_len, max_clip_last_one, current_clip_len = 0, 0, 0
        for i_clip in range(len(is_running)):
            if is_running[i_clip] == 1:
                current_clip_len += 1
                if current_clip_len > max_clip_len:
                    max_clip_len = current_clip_len
                    max_clip_last_one = i_clip
            else:
                current_clip_len = 0
        max_clip_first_one = max_clip_last_one - max_clip_len
        start_vicon = clip_len * max_clip_first_one + padding
        end_vicon = clip_len * (max_clip_last_one + 1) - padding
        return start_vicon, end_vicon

    @staticmethod
    def _initialize_path(processed_data_path, subject_folder):
        # create folder for this subject
        fre_100_path = processed_data_path + '\\' + subject_folder + '\\100Hz'
        fre_200_path = processed_data_path + '\\' + subject_folder + '\\200Hz'
        fre_1000_path = processed_data_path + '\\' + subject_folder + '\\1000Hz'
        if not os.path.exists(processed_data_path + '\\' + subject_folder):
            os.makedirs(processed_data_path + '\\' + subject_folder)
        if not os.path.exists(fre_100_path):
            os.makedirs(fre_100_path)
        if not os.path.exists(fre_200_path):
            os.makedirs(fre_200_path)
        if not os.path.exists(fre_1000_path):
            os.makedirs(fre_1000_path)
        return fre_100_path, fre_200_path, fre_1000_path

    @staticmethod
    def __save_data(folder_path, trial_name, data_all_df):
        data_file_str = '{folder_path}\\{trial_name}.csv'.format(folder_path=folder_path, trial_name=trial_name)
        data_all_df.to_csv(data_file_str, index=False)
