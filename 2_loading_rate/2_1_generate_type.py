"""
Generate the strike pattern, step rate of each step
"""
import numpy as np
from OneTrialData import OneTrialData
from const import SUB_NAMES, TRIAL_NAMES, MOCAP_SAMPLE_RATE, RAW_DATA_PATH, SI_SR_TRIALS
import xlrd
import pandas as pd


class OneTrialDataStepInfo(OneTrialData):

    def __init__(self, subject_name, trial_name, sensor_sampling_fre, strike_type_ends):
        super().__init__(subject_name, trial_name, sensor_sampling_fre, 1)
        trial_id = TRIAL_NAMES.index(trial_name)
        self.data_len = self.gait_data_df.shape[0]
        self.min_time_between_strike_off = int(sensor_sampling_fre * 0.15)

        self.sample_ranges = {}
        if 'SI' in trial_name:
            three_ends = np.array(strike_type_ends[trial_id])
            fore_start = self.__get_pattern_start(three_ends, 0)
            mid_start = self.__get_pattern_start(three_ends, 1)
            rear_start = self.__get_pattern_start(three_ends, 2)
            self.sample_ranges['forefoot'] = [fore_start, int(three_ends[0])]
            self.sample_ranges['midfoot'] = [mid_start, int(three_ends[1])]
            self.sample_ranges['rearfoot'] = [rear_start, int(three_ends[2])]

        elif 'SR' in trial_name:
            self.up_first = self.__if_up_first()
            self.sample_ranges['mid_0'] = [int(self.data_len * 1 / 6), int(self.data_len * 2 / 6)]
            self.sample_ranges['mid_1'] = [int(self.data_len * 4 / 6), int(self.data_len * 5 / 6)]
            if self.up_first:
                self.sample_ranges['hig_0'] = [0, int(self.data_len / 6)]
                self.sample_ranges['hig_1'] = [int(self.data_len * 5 / 6), self.data_len]
                self.sample_ranges['low_0'] = [int(self.data_len * 2 / 6), int(self.data_len * 3 / 6)]
                self.sample_ranges['low_1'] = [int(self.data_len * 3 / 6), int(self.data_len * 4 / 6)]
            else:
                self.sample_ranges['low_0'] = [0, int(self.data_len / 6)]
                self.sample_ranges['low_1'] = [int(self.data_len * 5 / 6), self.data_len]
                self.sample_ranges['hig_0'] = [int(self.data_len * 2 / 6), int(self.data_len * 3 / 6)]
                self.sample_ranges['hig_1'] = [int(self.data_len * 3 / 6), int(self.data_len * 4 / 6)]

    @staticmethod
    def __get_pattern_start(three_ends, index):
        fore_starts = three_ends[np.where(three_ends < three_ends[index])]
        if len(fore_starts) == 0:
            return 0
        else:
            return int(np.max(fore_starts))

    @staticmethod
    def __check_within_range(number, *ranges):
        for range in ranges:
            if range[0] < number < range[1]:
                return True
        return False

    def get_step_type(self, sample_num):
        if 'SI' in self._trial_name:
            if self.__check_within_range(sample_num, self.sample_ranges['forefoot']):
                return 'forefoot'
            elif self.__check_within_range(sample_num, self.sample_ranges['midfoot']):
                return 'midfoot'
            elif self.__check_within_range(sample_num, self.sample_ranges['rearfoot']):
                return 'rearfoot'

        elif 'SR' in self._trial_name:
            if self.__check_within_range(sample_num, self.sample_ranges['hig_0'], self.sample_ranges['hig_1']):
                return 'high_rate'
            elif self.__check_within_range(sample_num, self.sample_ranges['mid_0'], self.sample_ranges['mid_1']):
                return 'mid_rate'
            elif self.__check_within_range(sample_num, self.sample_ranges['low_0'], self.sample_ranges['low_1']):
                return 'low_rate'
        else:
            return None

    def __if_up_first(self):
        plate_strikes, step_num = self.get_strikes()
        step_len = [plate_strikes[i+1] - plate_strikes[i] for i in range(len(plate_strikes) - 1)]
        first_ten_step_len = np.median(step_len[:10])
        middle_ten_step_len = np.median(step_len[int(step_num/2)-5:int(step_num/2)+5])
        if first_ten_step_len < middle_ten_step_len:
            return True
        else:
            return False

    def get_info(self, imu_location='l_shank', from_IMU=2):
        filter_delay = 0
        if not from_IMU:
            offs, step_num = self.get_offs()
            strikes, _ = self.get_strikes()
        else:
            offs, strikes, step_num = self.get_offs_strikes_from_IMU(from_IMU)
        lr_data = self.gait_param_df[self._side + '_LR'].values
        IMU_data = self.get_multi_IMU_data_df([imu_location], gyr=True).values
        step_info, step_imu_data = [], []
        for i_step in range(step_num):
            strike_in_between = strikes[offs[i_step] < strikes]
            strike_in_between = strike_in_between[strike_in_between < offs[i_step+1]]
            if len(strike_in_between) != 1:
                continue
            step_start = offs[i_step] - filter_delay
            step_end = offs[i_step + 1] - filter_delay
            step_type = self.get_step_type(step_start)

            strikes_array = np.zeros([step_end - step_start, 1])
            strikes_array[strike_in_between - offs[i_step], 0] = 1
            step_input = np.column_stack([IMU_data[step_start:step_end, :], strikes_array])
            step_lr = lr_data[step_start:step_end]

            if step_end > lr_data.shape[0]:     # skip this step if the step_end exceeds the maximum data length
                continue
            if np.max(step_lr) <= 0:       # skip if there was no valid loading rate
                continue
            strikes_of_step = np.where(strikes_array == 1)[0]
            if len(strikes_of_step) != 1:        # delete steps without a valid strike time
                continue
            # delete steps if the duration between strike and off is too short
            stance_len = step_end - step_start - strikes_of_step[0]
            if not self.min_time_between_strike_off < stance_len:
                continue
            # delete steps if there is nan
            if np.isnan(step_input).any():
                continue

            step_imu_data.append(step_input)
            step_info.append([step_start, step_end, step_type])
        step_imu_data, step_info = self.check_step_input_output(step_imu_data, step_info)
        return step_imu_data, step_info


def export_predicted_values(predicted_value_df, test_date, test_name):
    predicted_value_df.columns = ['subject id', 'trial id', 'step start', 'step end', 'step type']
    file_path = 'result_conclusion/' + test_date + '/step_result/' + test_name + '.csv'
    predicted_value_df.to_csv(file_path, index=False)


def save_all_info(all_info_df, step_info, subject_name, trial_name):
    sub_id = SUB_NAMES.index(subject_name)
    trial_id = TRIAL_NAMES.index(trial_name)
    info_df = pd.DataFrame(step_info)
    info_df.insert(0, 'trial id', trial_id)
    info_df.insert(0, 'subject id', sub_id)
    all_info_df = all_info_df.append(info_df)
    return all_info_df



test_date = '1028'
all_info_df = pd.DataFrame()
for subject_name in SUB_NAMES:
    readme_xls = RAW_DATA_PATH + subject_name + '\\readme\\readme.xlsx'
    readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
    strike_type_ends = {}
    strike_type_ends[2] = readme_sheet.row_values(4)[8:11]
    strike_type_ends[5] = readme_sheet.row_values(7)[8:11]
    strike_type_ends[9] = readme_sheet.row_values(11)[8:11]
    strike_type_ends[12] = readme_sheet.row_values(14)[8:11]

    for trial_name in SI_SR_TRIALS:
        trial_reader = OneTrialDataStepInfo(subject_name, trial_name, MOCAP_SAMPLE_RATE, strike_type_ends)
        _, step_info = trial_reader.get_info()
        all_info_df = save_all_info(all_info_df, step_info, subject_name, trial_name)

all_info_df.columns = ['subject id', 'trial id', 'step start', 'step end', 'step type']
all_info_df = all_info_df.reset_index(drop=True)

file_path = 'result_conclusion/' + test_date + '/step_result/1028_step_info.csv'
all_info_df.to_csv(file_path, index=False)






















