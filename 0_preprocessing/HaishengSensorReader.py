import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from const import COLUMN_NAMES_HAISHENG, HAISHENG_SENSOR_SAMPLE_RATE, DATA_COLUMNS_IMU, DATA_COLUMNS_XSENS
import os
import xlrd
from shutil import copyfile
import scipy.interpolate as interpo
from IMUSensorReader import IMUSensorReader


class HaishengSensorReader(IMUSensorReader):
    def __init__(self, file, cut_off_fre=20, filter_order=4):
        super().__init__()
        self._file = file
        self.trial_name = file[file.rfind('\\'):]
        self.data_raw_df = self._get_raw_data()
        self.data_processed_df = self._get_sensor_data_processed(self.data_raw_df, cut_off_fre, filter_order)

    def _get_raw_data(self):
        """
        Interpolation was used to fill the missing package
        :return:
        """
        data_raw_df = pd.read_csv(self._file, usecols=range(13), header=None)
        data_raw_df.columns = COLUMN_NAMES_HAISHENG
        sample_num_col = HAISHENG_SENSOR_SAMPLE_RATE * (
                60 * data_raw_df['minute'] + data_raw_df['second'] + (data_raw_df['millisecond'] - 10) / 1000)
        data_raw_df.insert(0, 'sample', sample_num_col.astype(int))
        return data_raw_df

    def _get_sensor_data_processed(self, raw_data_df, cut_off_fre, filter_order):
        # remove duplicated samples
        cleaned_data_df = raw_data_df.drop_duplicates(subset='sample', keep='first', inplace=False)
        dropped_sample_num = raw_data_df.shape[0] - cleaned_data_df.shape[0]

        # interpolation and filtering
        wn = cut_off_fre / (HAISHENG_SENSOR_SAMPLE_RATE / 2)
        b_IMU, a_IMU = butter(filter_order, wn, btype='low')

        processed_data_df = pd.DataFrame()
        original_x = cleaned_data_df['sample'].values
        target_x = range(np.max(cleaned_data_df['sample']))

        for channel in DATA_COLUMNS_IMU:
            # interpolation
            original_y = cleaned_data_df[channel].values
            # For some unknown reason, Haisheng sensors' acceleration are always 10 times smaller than true value
            if 'acc' in channel:
                original_y = 10 * original_y
            interpo_f = interpo.interp1d(original_x, original_y, kind='linear')
            target_y = interpo_f(target_x)
            # filtering
            target_y = filtfilt(b_IMU, a_IMU, target_y)
            processed_data_df.insert(len(processed_data_df.columns), channel, target_y)
        # insert sample number
        processed_data_df.insert(0, 'sample', target_x)
        interpolated_sample_num = processed_data_df.shape[0] - cleaned_data_df.shape[0]
        print('{drop_num:d} samples dropped, {interpo_num:d} samples interpolated'.
              format(name=self.trial_name, drop_num=dropped_sample_num, interpo_num=interpolated_sample_num))
        return processed_data_df

    @staticmethod
    def rename_haisheng_sensor_files(sensor_folder, readme_xls):
        """
         This function copies and change the name of Haisheng's sensor file names.
        :param sensor_folder: str, the folder of Haisheng sensor data. Two more folder ('r_foot' and 'trunk') should
        be found in this folder.
        :param readme_xls: str, readme.xlsx from readme folder
        :return: None
        """
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        file_formal_names = readme_sheet.col_values(1)[2:]
        file_nums_foot = readme_sheet.col_values(2)[2:16]
        file_nums_trunk = readme_sheet.col_values(3)[2:16]
        sensor_locs = ['r_foot', 'trunk']
        file_nums_all = {sensor_locs[0]: file_nums_foot, sensor_locs[1]: file_nums_trunk}
        for sensor_loc in sensor_locs:
            # check if files have already been renamed
            folder_rename_str = '{path}\\{sensor_loc}_renamed'.format(path=sensor_folder, sensor_loc=sensor_loc)
            if os.path.exists(folder_rename_str):
                continue
            os.makedirs(folder_rename_str)

            # copy files
            file_nums_current_loc = file_nums_all[sensor_loc]
            for file_num, file_formal_name in zip(file_nums_current_loc, file_formal_names):
                file_ori_str = '{path}\\{sensor_loc}\\DATA_{file_num}.CSV'.format(
                    path=sensor_folder, sensor_loc=sensor_loc, file_num=int(file_num))
                file_new_str = '{rename_path}\\{file_formal_name}.csv'.format(
                    rename_path=folder_rename_str, file_formal_name=file_formal_name)
                copyfile(file_ori_str, file_new_str)

