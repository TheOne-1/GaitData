from const import COLUMN_FOR_HUAWEI_1000, PROCESSED_DATA_PATH, SUB_NAMES, SUB_AND_TRIALS, RUNNING_TRIALS
import os
import pandas as pd


COLUMN_FOR_LJX = ['l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z', 'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
                  'l_thigh_mag_x', 'l_thigh_mag_y', 'l_thigh_mag_z']

LXJ_DATA_PATH = 'D:\Tian\Research\Projects\HuaweiProject\SharedDocs\Huawei\PhaseIIData\DataForLJX'


def initialize_path(ljx_data_path, subject_folder):
    # create folder for this subject
    data_path_standing = ljx_data_path + '\\' + subject_folder + '\\standing'
    data_path_running = ljx_data_path + '\\' + subject_folder + '\\running'
    if not os.path.exists(data_path_standing):
        os.makedirs(data_path_standing)
    if not os.path.exists(data_path_running):
        os.makedirs(data_path_running)
    return data_path_standing, data_path_running



for subject_folder in SUB_NAMES:
    ori_200_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\200Hz'
    data_path_standing, data_path_running = initialize_path(LXJ_DATA_PATH, subject_folder)
    sub_trials = SUB_AND_TRIALS[subject_folder]

    for trial_name in RUNNING_TRIALS:
        gait_data_200_df = pd.read_csv(ori_200_path + '\\' + trial_name + '.csv', index_col=False)
        gait_data_200_df_hw = gait_data_200_df[COLUMN_FOR_LJX]

        data_file_str = '{folder_path}\\{trial_name}.csv'.format(
            folder_path=data_path_running, trial_name=trial_name)
        gait_data_200_df_hw.to_csv(data_file_str, index=False)

    for trial_name in ['nike static', 'mini static']:
        gait_data_200_df = pd.read_csv(ori_200_path + '\\' + trial_name + '.csv', index_col=False)
        gait_data_200_df_hw = gait_data_200_df[COLUMN_FOR_LJX]

        data_file_str = '{folder_path}\\{trial_name}.csv'.format(
            folder_path=data_path_standing, trial_name=trial_name)
        gait_data_200_df_hw.to_csv(data_file_str, index=False)




















