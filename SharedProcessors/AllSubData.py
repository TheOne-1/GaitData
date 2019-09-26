from const import PROCESSED_DATA_PATH, MOCAP_SAMPLE_RATE
from OneTrialData import OneTrialData, OneTrialDataStatic
from AllSubDataStruct import AllSubDataStruct


class AllSubData:

    def __init__(self, sub_and_trials, imu_locations, param_name, sensor_sampling_fre, strike_off_from_IMU=False):
        self._sub_and_trials = sub_and_trials  # subject names and corresponding trials in a dict
        self._sub_names = sub_and_trials.keys()
        self._param_name = param_name
        self._sensor_sampling_fre = sensor_sampling_fre
        self._strike_off_from_IMU = strike_off_from_IMU
        self.imu_locations = imu_locations

        # initialize the dataframe of gait data, including force plate, marker and IMU data
        self.__gait_data_path = PROCESSED_DATA_PATH + '\\'

    def get_all_data(self):
        all_sub_data_struct = AllSubDataStruct()
        for subject_name in self._sub_names:
            print('loading data of: ' + subject_name)
            static_nike_trial = OneTrialDataStatic(subject_name, 'nike static', self._sensor_sampling_fre)
            static_nike_df = static_nike_trial.get_multi_IMU_data(self.imu_locations, acc=True, mag=True)
            static_mini_trial = OneTrialDataStatic(subject_name, 'mini static', self._sensor_sampling_fre)
            static_mini_df = static_mini_trial.get_multi_IMU_data(self.imu_locations, acc=True, mag=True)
            trials = self._sub_and_trials[subject_name]
            for trial_name in trials:
                if 'nike' in trial_name:
                    trial_processor = OneTrialData(subject_name, trial_name, self._sensor_sampling_fre,
                                                   static_data_df=static_nike_df)
                else:
                    trial_processor = OneTrialData(subject_name, trial_name, self._sensor_sampling_fre,
                                                   static_data_df=static_mini_df)
                trial_input, trial_output = trial_processor.get_lr_input_output(
                    self.imu_locations, self._strike_off_from_IMU)
                all_sub_data_struct.append(trial_input, trial_output, subject_name, trial_name)
        return all_sub_data_struct
