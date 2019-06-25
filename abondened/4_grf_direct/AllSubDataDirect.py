from const import PROCESSED_DATA_PATH, MOCAP_SAMPLE_RATE, RAW_DATA_PATH
from OneTrialDataDirect import OneTrialDataDirect, OneTrialDataStatic
from AllSubDataStruct import AllSubDataStruct
import numpy as np


class AllSubData:

    def __init__(self, sub_and_trials, param_name, sensor_sampling_fre, strike_off_from_IMU=False):
        self._sub_and_trials = sub_and_trials  # subject names and corresponding trials in a dict
        self._sub_names = sub_and_trials.keys()
        self._param_name = param_name
        self._sensor_sampling_fre = sensor_sampling_fre
        self._strike_off_from_IMU = strike_off_from_IMU
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            self._side = 'l'  # 'l' or 'r'
        else:
            self._side = 'r'  # 'l' or 'r'

        # initialize the dataframe of gait data, including force plate, marker and IMU data
        self.__gait_data_path = PROCESSED_DATA_PATH + '\\'

    def get_all_data(self):
        all_input, all_output = np.zeros([0, 6]), np.zeros([0])
        for subject_name in self._sub_names:
            readme_xls = RAW_DATA_PATH + subject_name + '\\readme\\readme.xlsx'
            static_nike_trial = OneTrialDataStatic(subject_name, 'nike static', self._sensor_sampling_fre, readme_xls)
            static_nike_df = static_nike_trial.get_one_IMU_data(self._side + '_foot', acc=True, mag=True)
            static_mini_trial = OneTrialDataStatic(subject_name, 'mini static', self._sensor_sampling_fre, readme_xls)
            static_mini_df = static_mini_trial.get_one_IMU_data(self._side + '_foot', acc=True, mag=True)
            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                if 'nike' in trial_name:
                    trial_processor = OneTrialDataDirect(subject_name, trial_name, self._sensor_sampling_fre, readme_xls,
                                                         static_data_df=static_nike_df)
                else:
                    trial_processor = OneTrialDataDirect(subject_name, trial_name, self._sensor_sampling_fre, readme_xls,
                                                         static_data_df=static_mini_df)
                all_input = np.row_stack([all_input, trial_processor.get_one_IMU_data(self._side + '_foot', gyr=True)])
                all_output = np.concatenate([all_output, trial_processor.get_GRF_data()])
        return all_input, all_output
