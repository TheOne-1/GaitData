from const import COLUMN_FOR_HUAWEI_1000, PROCESSED_DATA_PATH, SUB_NAMES, SUB_AND_TRIALS, HUAWEI_DATA_PATH, COLUMN_FOR_HUAWEI
import os
import pandas as pd
from shutil import copyfile


def initialize_path(huawei_data_path, subject_folder):
    # create folder for this subject
    fre_200_path = huawei_data_path + '\\' + subject_folder + '\\200Hz'
    fre_1000_path = huawei_data_path + '\\' + subject_folder + '\\1000Hz'
    if not os.path.exists(huawei_data_path + '\\' + subject_folder):
        os.makedirs(huawei_data_path + '\\' + subject_folder)
    if not os.path.exists(fre_200_path):
        os.makedirs(fre_200_path)
    if not os.path.exists(fre_1000_path):
        os.makedirs(fre_1000_path)
    return fre_200_path, fre_1000_path


for subject_folder in SUB_NAMES:
    ori_200_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\200Hz'
    ori_1000_path = PROCESSED_DATA_PATH + '\\' + subject_folder + '\\1000Hz'
    fre_200_path, fre_1000_path = initialize_path(HUAWEI_DATA_PATH, subject_folder)
    sub_trials = SUB_AND_TRIALS[subject_folder]

    for trial_name in sub_trials:
        # copy 200 Hz data
        gait_data_200_df = pd.read_csv(ori_200_path + '\\' + trial_name + '.csv', index_col=False)
        gait_data_200_df_hw = gait_data_200_df[COLUMN_FOR_HUAWEI]

        data_file_str = '{folder_path}\\{trial_name}.csv'.format(
            folder_path=fre_200_path, trial_name=trial_name)
        gait_data_200_df_hw.to_csv(data_file_str, index=False)

        # copy 1000 Hz data
        gait_data_1000_df = pd.read_csv(ori_1000_path + '\\' + trial_name + '.csv', index_col=False)
        gait_data_1000_df_hw = gait_data_1000_df[COLUMN_FOR_HUAWEI_1000]

        data_file_str = '{folder_path}\\{trial_name}.csv'.format(
            folder_path=fre_1000_path, trial_name=trial_name)
        gait_data_1000_df_hw.to_csv(data_file_str, index=False)





















