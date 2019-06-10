from const import PROCESSED_DATA_PATH, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE, ROTATION_VIA_STATIC_CALIBRATION,\
    SPECIFIC_CALI_MATRIX, FILTER_BUFFER
import numpy as np
import pandas as pd
import xlrd


class OneTrialDataDirect:
    def __init__(self, subject_name, trial_name, sensor_sampling_fre, readme_xls, static_data_df=None):
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        self.__weight = readme_sheet.cell_value(17, 1)  # in kilos
        self._subject_name = subject_name
        self._trial_name = trial_name
        self._sensor_sampling_fre = sensor_sampling_fre
        self._static_data_df = static_data_df
        if sensor_sampling_fre == MOCAP_SAMPLE_RATE:
            self._side = 'l'       # 'l' or 'r'
            data_folder = '\\200Hz\\'
        else:
            self._side = 'r'       # 'l' or 'r'
            data_folder = '\\100Hz\\'
        # initialize the dataframe of gait data, including force plate, marker and IMU data
        gait_data_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + trial_name + '.csv'
        self.gait_data_df = pd.read_csv(gait_data_path, index_col=False)
        # initialize the dataframe of gait parameters, including loading rate, strike index, ...
        gait_param_path = PROCESSED_DATA_PATH + '\\' + subject_name + data_folder + 'param_of_' + trial_name + '.csv'
        buffer_sample_num = self._sensor_sampling_fre * FILTER_BUFFER
        self.gait_data_df = self.gait_data_df.loc[buffer_sample_num:, :]        # skip the first several hundred data
        if static_data_df is not None:
            self.gait_param_df = pd.read_csv(gait_param_path, index_col=False)
            self.gait_param_df = self.gait_param_df.loc[buffer_sample_num:, :]

    def get_GRF_data(self):
        grf_z = self.gait_data_df['f_1_z'].values
        data_len = grf_z.shape[0]
        r_strikes = self.gait_param_df['r_strikes']
        r_offs = self.gait_param_df['r_offs']
        diffs = r_strikes - r_offs
        return grf_z / self.__weight

    def get_one_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]

        data = self.gait_data_df[column_names].values
        if ROTATION_VIA_STATIC_CALIBRATION:
            data_rotated = np.zeros(data.shape)
            if self._subject_name in SPECIFIC_CALI_MATRIX.keys() and\
                    IMU_location in SPECIFIC_CALI_MATRIX[self._subject_name].keys():
                    dcm_mat = SPECIFIC_CALI_MATRIX[self._subject_name][IMU_location]
            else:
                dcm_mat = self.get_rotation_via_static_cali(IMU_location)
            data_len = data.shape[0]
            for i_sample in range(data_len):
                if acc:
                    data_rotated[i_sample, 0:3] = np.matmul(dcm_mat, data[i_sample, 0:3])
                if gyr:
                    data_rotated[i_sample, 3:6] = np.matmul(dcm_mat, data[i_sample, 3:6])
                if mag:
                    data_rotated[i_sample, 6:9] = np.matmul(dcm_mat, data[i_sample, 6:9])
            return data_rotated
        else:
            return data

    def get_rotation_via_static_cali(self, IMU_location):
        axis_name_gravity = [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        data_gravity = self._static_data_df[axis_name_gravity]
        vector_gravity = np.mean(data_gravity.values, axis=0)

        axis_name_mag = [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        try:
            data_mag = self._static_data_df[axis_name_mag]
        except KeyError:
            pass
        vector_mag = np.mean(data_mag.values, axis=0)

        vector_2 = vector_gravity / np.linalg.norm(vector_gravity)
        vector_0 = np.cross(vector_mag, vector_gravity)
        vector_0 = vector_0 / np.linalg.norm(vector_0)
        vector_1 = np.cross(vector_2, vector_0)
        vector_1 = vector_1 / np.linalg.norm(vector_1)

        dcm_mat = np.array([vector_0, vector_1, vector_2])
        return dcm_mat


class OneTrialDataStatic(OneTrialDataDirect):
    def get_one_IMU_data(self, IMU_location, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        data = self.gait_data_df[column_names]
        return data


