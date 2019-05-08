from Processor import TrialProcessor
from const import FILE_NAMES
import numpy as np
from Evaluation import Evaluation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class ProcessorLR:
    def __init__(self, subject_name, side, sensor_sampling_fre):
        self._x_train, self._y_train = ProcessorLR.get_all_data_list(
            subject_name, side, sensor_sampling_fre, FILE_NAMES[2:3])
        self._x_test, self._y_test = ProcessorLR.get_all_data_list(
            subject_name, side, sensor_sampling_fre, FILE_NAMES[2:3])

    def linear_regression_solution(self):
        model = LinearRegression()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'strike index')
        plt.show()

    @staticmethod
    def get_all_data_list(subject_name, side, sensor_sampling_fre, trials):
        input_all_list, output_all_list = [], []
        for trial_name in trials:
            trial_processor = TrialProcessor(subject_name, trial_name, side, sensor_sampling_fre)
            trial_input = trial_processor.get_step_IMU_data('r_foot', gyr=True)
            input_all_list.extend(trial_input)
            trial_output = trial_processor.get_step_param('LR')
            output_all_list.extend(trial_output)
        input_all_list, output_all_list = ProcessorLR.drop_illegal_LR_sample(input_all_list, output_all_list)
        input_all_array = ProcessorLR.convert_input(input_all_list)
        output_all_array = ProcessorLR.convert_output(output_all_list)
        return input_all_array, output_all_array

    @staticmethod
    def convert_output(output_all_list):
        step_num = len(output_all_list)
        step_output = np.zeros([step_num])
        for i_step in range(step_num):
            step_output[i_step] = np.max(output_all_list[i_step])
        return step_output.reshape(-1, 1)

    # convert the input from list to ndarray
    @staticmethod
    def convert_input(input_all_list, start_phase_0=0, end_phase_0=0.12,
                            start_phase_1=0.62, end_phase_1=0.75,
                            start_phase_2=0, end_phase_2=0.08):
        step_num = len(input_all_list)
        step_input = np.zeros([step_num, 3])
        for i_step in range(step_num):
            acc_data = input_all_list[i_step][:, 0:3]
            gyr_data = input_all_list[i_step][:, 3:6]
            step_len = input_all_list[i_step].shape[0]
            acc_normed = np.linalg.norm(acc_data, axis=1)

            # feature 0, std of acc after heel strike
            start_sample = int(round(start_phase_0 * step_len))
            end_sample = int(round(end_phase_0 * step_len))
            step_input[i_step, 0] = np.std(acc_normed[start_sample:end_sample])

            # feature 1, rotation after heel strike
            start_sample = int(round(start_phase_0 * step_len))
            end_sample = int(round(end_phase_0 * step_len))
            gyr_x = gyr_data[:, 0]
            roll_rotation = np.mean(gyr_x[start_sample:end_sample])
            step_input[i_step, 1] = roll_rotation

            # feature 2, max acc
            start_sample = int(round(start_phase_2 * step_len))
            end_sample = int(round(end_phase_2 * step_len))
            acc_slice = acc_normed[start_sample:end_sample]
            acc_step = np.arange(len(acc_slice))
            acc_step_resampled = np.arange(0, len(acc_slice)-1, 0.02)
            z = np.polyfit(acc_step, acc_slice, 3)
            p = np.poly1d(z)
            acc_slice_fit = p(acc_step_resampled)

            # plt.plot(acc_step, acc_slice, 'b')
            # plt.plot(acc_step_resampled, acc_slice_fit, 'r')

            step_input[i_step, 2] = np.max(acc_slice_fit)
        # plt.show()
        return step_input

    @staticmethod
    def drop_illegal_LR_sample(input, output):
        i_step = 0
        while i_step < len(input):
            if np.max(output[i_step]) == 0:
                input.pop(i_step)
                output.pop(i_step)
            i_step += 1
        return input, output


