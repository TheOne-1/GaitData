
from const import PROCESSED_DATA_PATH, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE
import numpy as np
import pandas as pd
from OneTrialData import OneTrialData
from AllSubDataStruct import AllSubDataStruct


class AllSubData:

    def __init__(self, sub_and_trials, param_name, sensor_sampling_fre, strike_off_from_IMU=False):
        self._sub_and_trials = sub_and_trials      # subject names and corresponding trials in a dict
        self._sub_names = sub_and_trials.keys()
        self._param_name = param_name
        self._sensor_sampling_fre = sensor_sampling_fre
        self._strike_off_from_IMU = strike_off_from_IMU
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            data_folder = '\\200Hz\\'
            self._side = 'l'       # 'l' or 'r'
        else:
            data_folder = '\\100Hz\\'
            self._side = 'r'       # 'l' or 'r'

        # initialize the dataframe of gait data, including force plate, marker and IMU data
        self.__gait_data_path = PROCESSED_DATA_PATH + '\\'

    def get_all_data(self):
        all_sub_data_struct = AllSubDataStruct()
        for subject_name in self._sub_names:
            # static_nike_trial = OneTrialData(subject_name, 'nike static', self._sensor_sampling_fre)
            # static_mini_trial = OneTrialData(subject_name, 'mini static', self._sensor_sampling_fre)
            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                trial_processor = OneTrialData(subject_name, trial_name, self._sensor_sampling_fre)
                trial_input = trial_processor.get_step_strike_and_IMU_data(
                    self._side + '_foot', gyr=True, from_IMU=self._strike_off_from_IMU)
                trial_output = trial_processor.get_step_param('LR', from_IMU=self._strike_off_from_IMU)
                all_sub_data_struct.append(trial_input, trial_output, subject_name, trial_name)
        return all_sub_data_struct


