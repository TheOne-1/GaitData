from keras.layers import *
from ProcessorLR import ProcessorLR


class ProcessorLRSklearn(ProcessorLR):
    # convert the input from list to ndarray
    @staticmethod
    def convert_input(input_all_list, sampling_fre):
        """
        Min, max, sum feature based solution
        """
        step_num = len(input_all_list)
        step_input = np.zeros([step_num, 13])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:6]
            acc_gyr_data_resampled = np.zeros([100, 6])
            for i_channel in range(6):
                acc_gyr_data_resampled[:, i_channel] = ProcessorLR.resample_channel(acc_gyr_data[:, i_channel], 100)
            strike_data = input_all_list[i_step][:, 6]
            step_len = input_all_list[i_step].shape[0]
            acc_normed = np.linalg.norm(acc_gyr_data[:, 0:3], axis=1)

            # feature 0, strike time
            strike_sample_num = np.where(strike_data == 1)[0][0]
            strike_sample_num_percent = strike_sample_num / step_len
            step_input[i_step, 0] = strike_sample_num_percent

            # feature 1, step length
            step_input[i_step, 1] = step_len

            # feature 2, acc_z at heel strike
            step_input[i_step, 2] = np.mean(acc_gyr_data[strike_sample_num, 2])

            # feature 3, acc_z 52 - 55
            start_phase_3, end_phase_3 = 52, 55
            step_input[i_step, 3] = np.mean(acc_gyr_data_resampled[start_phase_3:end_phase_3, 2])
            step_input[i_step, 8] = np.std(acc_gyr_data_resampled[start_phase_3:end_phase_3, 2])

            # feature 4, gyr_y 53 - 57
            start_phase_4, end_phase_4 = 53, 57
            step_input[i_step, 4] = np.mean(acc_gyr_data_resampled[start_phase_4:end_phase_4, 4])
            step_input[i_step, 9] = np.std(acc_gyr_data_resampled[start_phase_4:end_phase_4, 4])

            # feature 5, gyr_y 64 - 66
            start_phase_5, end_phase_5 = 64, 66
            step_input[i_step, 5] = np.mean(acc_gyr_data_resampled[start_phase_5:end_phase_5, 4])
            step_input[i_step, 10] = np.std(acc_gyr_data_resampled[start_phase_5:end_phase_5, 4])

            # feature 6, gyr_z 53 - 57
            start_phase_6, end_phase_6 = 42, 48
            step_input[i_step, 6] = np.mean(acc_gyr_data_resampled[start_phase_6:end_phase_6, 5])
            step_input[i_step, 11] = np.std(acc_gyr_data_resampled[start_phase_6:end_phase_6, 5])

            # feature 7, gyr_x 50 - 60
            start_phase_7, end_phase_7 = 50, 60
            step_input[i_step, 7] = np.mean(acc_gyr_data_resampled[start_phase_7:end_phase_7, 3])
            step_input[i_step, 12] = np.std(acc_gyr_data_resampled[start_phase_7:end_phase_7, 3])

        feature_names = ['strike_sample_num_percent', 'step length', 'acc_z at heel strike', 'acc_z 52 - 55',
                         'gyr_y 53 - 57', 'gyr_y 64 - 66', 'gyr_z 53 - 57', 'gyr_x 50 - 60']
        return step_input, feature_names

