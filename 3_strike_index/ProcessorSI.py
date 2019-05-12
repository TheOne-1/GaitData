from Processor import TrialProcessor
from const import FILE_NAMES
import numpy as np
from Evaluation import Evaluation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class ProcessorSI:
    def __init__(self, subject_name, side, sensor_sampling_fre):
        self._x_train, self._y_train = ProcessorSI.get_all_data(
            subject_name, side, sensor_sampling_fre, FILE_NAMES[1:7])
        self._x_test, self._y_test = ProcessorSI.get_all_data(
            subject_name, side, sensor_sampling_fre, FILE_NAMES[1:7])

    def linear_regression_solution(self):
        model = LinearRegression()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    @staticmethod
    def get_all_data(subject_name, side, sensor_sampling_fre, trials):
        input_all_list, output_all_list, strikes_all_list = [], [], []
        for trial_name in trials:
            trial_processor = TrialProcessor(subject_name, trial_name, side, sensor_sampling_fre)
            trial_input = trial_processor.get_step_IMU_data('r_foot', gyr=True)
            input_all_list.extend(trial_input)
            trial_output = trial_processor.get_step_param('strike_index')
            output_all_list.extend(trial_output)
            trial_strikes = trial_processor.get_step_param('strikes')
            strikes_all_list.extend(trial_strikes)
        input_all_array = ProcessorSI.convert_input(input_all_list)
        output_all_array = ProcessorSI.convert_output(output_all_list, strikes_all_list)
        return input_all_array, output_all_array

    @staticmethod
    def convert_output(output_all_list, trial_strikes, sample_delay=4):
        step_num = len(output_all_list)
        step_output = np.zeros([step_num])
        for i_step in range(step_num):
            step_strike_all = trial_strikes[i_step]
            if len(np.where(step_strike_all == 1)[0]) != 0:
                step_strike_sample = np.where(step_strike_all == 1)[0][0]
            else:
                step_strike_sample = 41
            step_output[i_step] = output_all_list[i_step][step_strike_sample + sample_delay]
        return step_output.reshape(-1, 1)

    # convert the input from list to ndarray
    @staticmethod
    def convert_input(input_all_list, start_phase_0=0, end_phase_0=0.08, start_phase_1=0.55, end_phase_1=0.65):
        step_num = len(input_all_list)
        step_input = np.zeros([step_num, 3])
        for i_step in range(step_num):
            # feature 0, rotation after heel strike
            gyr_data = input_all_list[i_step][:, 3:6]
            step_len = gyr_data.shape[0]
            start_sample = int(round(start_phase_0 * step_len))
            end_sample = int(round(end_phase_0 * step_len))
            gyr_x = gyr_data[:, 0]
            roll_rotation = np.mean(gyr_x[start_sample:end_sample])
            step_input[i_step, 0] = roll_rotation

            # feature 1, acc_y over acc_norm
            acc_data = input_all_list[i_step][:, 0:3]
            start_sample = int(round(start_phase_1 * step_len))
            end_sample = int(round(end_phase_1 * step_len))
            acc_y = acc_data[start_sample:end_sample, 1]
            acc_norm = np.linalg.norm(acc_data[start_sample:end_sample, :], axis=1)
            acc_y_to_norm = np.mean(acc_y / acc_norm)
            step_input[i_step, 1] = acc_y_to_norm

        return step_input
