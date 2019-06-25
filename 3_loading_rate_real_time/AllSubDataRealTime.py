from const import PROCESSED_DATA_PATH, MOCAP_SAMPLE_RATE
from OneTrialDataRealTime import OneTrialDataRealTime, OneTrialStatic
from AllSubDataStructRealTime import AllSubDataStructRealTime
import numpy as np
import matplotlib.pyplot as plt


class AllSubDataRealTime:

    def __init__(self, sub_and_trials, param_name, sensor_sampling_fre, strike_off_from_IMU=False):
        self._sub_and_trials = sub_and_trials  # subject names and corresponding trials in a dict
        self._sub_names = list(sub_and_trials.keys())
        self._param_name = param_name
        self._sensor_sampling_fre = sensor_sampling_fre
        self._strike_off_from_IMU = strike_off_from_IMU
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            self._side = 'l'  # 'l' or 'r'
        else:
            self._side = 'r'  # 'l' or 'r'

        # initialize the dataframe of gait data, including force plate, marker and IMU data
        self.__gait_data_path = PROCESSED_DATA_PATH + '\\'

    def get_all_data(self, clean=True):
        all_sub_data_struct = AllSubDataStructRealTime()
        for subject_name in self._sub_names:
            static_nike_trial = OneTrialStatic(subject_name, 'nike static', self._sensor_sampling_fre)
            static_nike_df = static_nike_trial.get_one_IMU_data(self._side + '_foot', acc=True, mag=True)
            static_mini_trial = OneTrialStatic(subject_name, 'mini static', self._sensor_sampling_fre)
            static_mini_df = static_mini_trial.get_one_IMU_data(self._side + '_foot', acc=True, mag=True)
            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                if 'nike' in trial_name:
                    trial_processor = OneTrialDataRealTime(subject_name, trial_name, self._sensor_sampling_fre,
                                                           static_data_df=static_nike_df)
                else:
                    trial_processor = OneTrialDataRealTime(subject_name, trial_name, self._sensor_sampling_fre,
                                                           static_data_df=static_mini_df)
                inertial_input, aux_input, output = trial_processor.get_trial_data()
                all_sub_data_struct.append(inertial_input, aux_input, output, subject_name, trial_name)

                # # !!! just for debug
                # for step_input in inertial_input:
                #     plt.plot(step_input[:, 3])
                # plt.show()
        if clean:
            all_sub_data_struct = AllSubDataRealTime.__clean_all_data(all_sub_data_struct)
        return all_sub_data_struct

    @staticmethod
    def __clean_all_data(all_sub_data_struct):
        i_step = 0
        input_list, aux_input_list, output_list = all_sub_data_struct.get_input_output_list()
        while i_step < len(all_sub_data_struct):
            # delete steps without a valid strike occurrence sample number
            strike_sample_num = aux_input_list[i_step][1]
            if strike_sample_num <= 10:
                all_sub_data_struct.pop(i_step)
            # delete steps without a valid strike time
            else:
                # step number only increase when no pop happens
                i_step += 1
        return all_sub_data_struct

















