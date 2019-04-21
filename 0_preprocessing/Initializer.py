import pandas as pd
import numpy as np
from const import RAW_DATA_PATH, FILE_NAMES, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE, \
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
    def __init__(self, processed_data_path, subject_folder, readme_xls, check_sync=True, initialize_100Hz=True,
                 initialize_200Hz=True):
        self.__processed_data_path = processed_data_path
        self._subject_folder = subject_folder
        self.__readme_xls = readme_xls

        fre_100_path, fre_200_path = SubjectDataInitializer.__initialize_path(processed_data_path, subject_folder)

        # initialize 100 Hz data
        if initialize_100Hz:
            for trial_name in FILE_NAMES[1:]:
                print('Initializing {trial_name} trial, vicon and xsens, 200 Hz...'.format(trial_name=trial_name))
                vicon_all_df, l_foot_marker_df, r_foot_marker_df, start_vicon, end_vicon = self.initialize_vicon_resampled(
                    trial_name, HAISHENG_SENSOR_SAMPLE_RATE)
                haisheng_df = self.initialize_haisheng_sensor(
                    trial_name, 'r_foot', r_foot_marker_df, start_vicon, end_vicon)
                SubjectDataInitializer.__save_data(fre_100_path, trial_name, vicon_all_df, haisheng_df)
                if check_sync:
                    self.check_sync(trial_name, vicon_all_df, haisheng_df, 'r_foot', HAISHENG_SENSOR_SAMPLE_RATE)
            plt.show()

        # initialize 200 Hz data
        if initialize_200Hz:
            for trial_name in FILE_NAMES:
                print('Initializing {trial_name} trial, vicon and xsens, 200 Hz...'.format(trial_name=trial_name))
                vicon_all_df, l_foot_marker_df, r_foot_marker_df, start_vicon, end_vicon = self.initialize_vicon(trial_name)
                xsens_all_df = self.initialize_xsens(trial_name, l_foot_marker_df, start_vicon, end_vicon)
                SubjectDataInitializer.__save_data(fre_200_path, trial_name, vicon_all_df, xsens_all_df)
                if check_sync:
                    self.check_sync(trial_name, vicon_all_df, xsens_all_df, 'l_foot')
            plt.show()

    def initialize_haisheng_sensor(self, trial_name, location, marker_df, start_vicon, end_vicon):
        file_path_haisheng = '{path}{sub_folder}\\{sensor}\\{sensor_loc}_renamed\\{trial_name}.csv'.format(
                path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='haisheng', sensor_loc=location,
                trial_name=trial_name)
        haisheng_sensor_reader = HaishengSensorReader(file_path_haisheng)
        sensor_gyr_norm = haisheng_sensor_reader.get_normalized_gyr()

        # get gyr norm from simulation
        my_nike_gyr_simulator = GyrSimulator(self._subject_folder, location)
        gyr_vicon = my_nike_gyr_simulator.get_gyr(trial_name, marker_df, sampling_rate=HAISHENG_SENSOR_SAMPLE_RATE)
        gyr_norm_vicon = norm(gyr_vicon, axis=1)

        vicon_delay = GyrSimulator.sync_vicon_sensor(trial_name, 'r_foot', gyr_norm_vicon, sensor_gyr_norm, check=False)
        start_haisheng, end_haisheng = start_vicon + vicon_delay, end_vicon + vicon_delay
        haisheng_sensor_df = haisheng_sensor_reader.data_processed_df.copy().loc[start_haisheng:end_haisheng]
        haisheng_sensor_df = haisheng_sensor_df.reset_index(drop=True)
        return haisheng_sensor_df

    def initialize_xsens(self, trial_name, l_foot_marker_df, start_vicon, end_vicon, check=False):
        # get gyr norm from left foot xsens sensor
        file_path_xsens = '{path}{sub_folder}\\{sensor}\\{trial_folder}\\'.format(
            path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='xsens', trial_folder=trial_name)
        l_foot_xsens_reader = XsensReader(file_path_xsens + XSENS_FILE_NAME_DIC['l_foot'])
        sensor_gyr_norm = l_foot_xsens_reader.get_normalized_gyr()
        # get gyr norm from simulation
        my_nike_gyr_simulator = GyrSimulator(self._subject_folder, 'l_foot')
        gyr_vicon = my_nike_gyr_simulator.get_gyr(trial_name, l_foot_marker_df,
                                                  sampling_rate=MOCAP_SAMPLE_RATE)
        gyr_norm_vicon = norm(gyr_vicon, axis=1)
        vicon_delay = GyrSimulator.sync_vicon_sensor(trial_name, 'l_foot', gyr_norm_vicon, sensor_gyr_norm, check)
        start_xsens, end_xsens = start_vicon + vicon_delay, end_vicon + vicon_delay

        xsens_all_df = pd.DataFrame()
        for xsens_location in XSENS_SENSOR_LOACTIONS:
            current_xsens_col_names = [xsens_location + '_' + channel for channel in DATA_COLUMNS_IMU]
            xsens_reader = XsensReader(file_path_xsens + XSENS_FILE_NAME_DIC[xsens_location])
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
                vicon_reader.get_plate_data_resampled(), self.__readme_xls, trial_name)
        else:  # baseline or strike
            start_vicon, end_vicon = self.__find_running_period(vicon_reader.marker_data_processed_df['LFCC_y'],
                                                                running_thd=300)
        if check_running_period:
            f_1_z_data = vicon_reader.get_plate_data_resampled()['f_1_z']
            plt.plot(f_1_z_data)
            plt.plot([start_vicon, start_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'g--')
            plt.plot([end_vicon, end_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'r--')
            plt.show()

        vicon_all_df = vicon_reader.get_vicon_all_processed_df()
        vicon_all_df = vicon_all_df.loc[start_vicon:end_vicon].reset_index(drop=True)
        l_foot_marker_df = vicon_reader.get_marker_data_processed_segment('l_foot')
        r_foot_marker_df = vicon_reader.get_marker_data_processed_segment('r_foot')
        return vicon_all_df, l_foot_marker_df, r_foot_marker_df, start_vicon, end_vicon

    def initialize_vicon_resampled(self, trial_name, sampling_rate, check_running_period=False):
        file_path_vicon = '{path}{sub_folder}\\{sensor}\\{file_name}.csv'.format(
            path=RAW_DATA_PATH, sub_folder=self._subject_folder, sensor='vicon', file_name=trial_name)
        vicon_reader = ViconReader(file_path_vicon)
        # self.plot_trajectory(vicon_reader.marker_data_processed_df['LFCC_y'])

        if 'static' in trial_name:
            # 4 second preparation time
            start_vicon, end_vicon = 5 * MOCAP_SAMPLE_RATE, STATIC_STANDING_PERIOD * MOCAP_SAMPLE_RATE
        elif 'SI' in trial_name:
            start_vicon, end_vicon = self.__find_three_pattern_period(
                vicon_reader.get_plate_data_resampled(), self.__readme_xls, trial_name)
        else:           # baseline or strike
            start_vicon, end_vicon = self.__find_running_period(vicon_reader.marker_data_processed_df['LFCC_y'],
                                                                running_thd=300)
        if check_running_period:
            f_1_z_data = vicon_reader.get_plate_data_resampled()['f_1_z']
            plt.plot(f_1_z_data)
            plt.plot([start_vicon, start_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'g--')
            plt.plot([end_vicon, end_vicon], [np.min(f_1_z_data), np.max(f_1_z_data)], 'r--')
            plt.show()

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
        segment_marker_names = SEGMENT_MARKERS[location]
        segment_marker_names_xyz = [name + axis for name in segment_marker_names for axis in ['_x', '_y', '_z']]
        marker_df_clip = marker_df[segment_marker_names_xyz].copy().reset_index(drop=True).loc[0:check_len]
        if location is 'l_foot':
            gyr_column_names = ['l_foot_gyr_' + axis for axis in ['x', 'y', 'z']]
            sensor_df = sensor_df[gyr_column_names].loc[:check_len]
        elif location is 'r_foot':
            sensor_df = sensor_df[['gyr_x', 'gyr_y', 'gyr_z']].loc[:check_len]
        elif location is 'trunk':
            pass
        else:
            raise ValueError('Wrong sensor location')

        # get gyr norm from simulation
        my_nike_gyr_simulator = GyrSimulator(self._subject_folder, location)
        gyr_vicon = my_nike_gyr_simulator.get_gyr(trial_name, marker_df_clip, sampling_rate=HAISHENG_SENSOR_SAMPLE_RATE)
        gyr_norm_vicon = norm(gyr_vicon, axis=1)
        gyr_norm_sensor = norm(sensor_df, axis=1)
        plt.figure()
        plt.plot(gyr_norm_vicon)
        plt.plot(gyr_norm_sensor)

    @staticmethod
    def plot_trajectory(marker_df):
        # just for testing
        plt.plot(marker_df)
        plt.show()

    @staticmethod
    def __find_three_pattern_period(plate_df, readme_xls, trial_name, force_thd=200):
        # find the start time when subject stepped on the first force plate
        f_1_z = abs(plate_df['f_1_z'].values)
        start_vicon = np.argmax(f_1_z > force_thd)

        # find the end of three patterns from readme
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        trial_num = 0
        for sheet_trial_name in readme_sheet.col_values(1)[2:]:
            if sheet_trial_name == trial_name:
                break
            trial_num += 1
        pattern_ends = readme_sheet.row_values(trial_num+2)[8:11]
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
    def __initialize_path(processed_data_path, subject_folder):
        # create folder for this subject
        fre_100_path = processed_data_path + '\\' + subject_folder + '\\100Hz'
        fre_200_path = processed_data_path + '\\' + subject_folder + '\\200Hz'
        if not os.path.exists(processed_data_path + '\\' + subject_folder):
            os.makedirs(processed_data_path + '\\' + subject_folder)
        if not os.path.exists(fre_100_path):
            os.makedirs(fre_100_path)
        if not os.path.exists(fre_200_path):
            os.makedirs(fre_200_path)
        return fre_100_path, fre_200_path

    @staticmethod
    def __save_data(folder_path, trial_name, vicon_df, sensor_df):
        data_all_df = pd.concat([vicon_df, sensor_df], axis=1)
        data_file_str = '{folder_path}\\{trial_name}.csv'.format(folder_path=folder_path, trial_name=trial_name)
        data_all_df.to_csv(data_file_str, index=False)
        x=1
















