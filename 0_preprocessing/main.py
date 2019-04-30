import numpy as np
import ViconReader
import HaishengSensorReader
import XsensReader
from const import FOLDER_PATH, FILE_NAMES
import matplotlib.pyplot as plt
import sklearn.svm as svm


subject_folder = '190414WangDianxin'
file_path_xsens = '{path}{sub_folder}\\{sensor}\\{trial_folder}\\'.format(
    path=FOLDER_PATH, sub_folder=subject_folder, sensor='xsens', trial_folder=FILE_NAMES[0])
file_path_vicon = '{path}{sub_folder}\\{sensor}\\{file_name}.csv'.format(
    path=FOLDER_PATH, sub_folder=subject_folder, sensor='vicon', file_name=FILE_NAMES[0])
file_path_haisheng = '{path}{sub_folder}\\{sensor}\\{sensor_loc}\\{trial_name}.csv'.format(
    path=FOLDER_PATH, sub_folder=subject_folder, sensor='haisheng', sensor_loc='foot_renamed', trial_name=FILE_NAMES[0])
readme_xls_path = FOLDER_PATH + subject_folder + '\\readme\\readme.xlsx'

my_xsens_reader = XsensReader(file_path_xsens+'MT_03700647_000.mtb')
data = my_xsens_reader.get_channel_data_processed('acc_x')


HaishengSensorReader.rename_haisheng_sensor_files(FOLDER_PATH + subject_folder + '\\haisheng', readme_xls_path)
my_haisheng_sensor_reader = HaishengSensorReader(file_path_haisheng)
gyr_1 = my_haisheng_sensor_reader._get_channel_data_raw('gyr_z')

my_vicon_reader = ViconReader(file_path_vicon)
segment_marker_df = my_vicon_reader.get_marker_data_processed_segment('r_foot')
segment_marker_df_resampled = my_vicon_reader.get_plate_data_resampled()
data_marker_segment_mat = segment_marker_df.values

