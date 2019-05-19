from Evaluation import Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from AllSubData import AllSubData
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import plot_model
from matplotlib import colors as mcolors
from const import SUB_NAMES, TRIAL_NAMES, COLORS


class ProcessorLR:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=False):
        param_name = 'LR'
        train_all_data = AllSubData(train_sub_and_trials, param_name, sensor_sampling_fre, strike_off_from_IMU)
        train_all_data_list = train_all_data.get_all_data()
        train_all_data_list = ProcessorLR.clean_all_data(train_all_data_list)
        input_list, output_list = train_all_data_list.get_input_output_list()
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()
        self._x_train, feature_names = ProcessorLR.convert_input_0(input_list, sensor_sampling_fre)
        self._y_train = ProcessorLR.convert_output(output_list)
        ProcessorLR.draw_correlation(self._x_train, self._y_train, sub_id_list, SUB_NAMES, feature_names)
        ProcessorLR.draw_correlation(self._x_train, self._y_train, trial_id_list, TRIAL_NAMES, feature_names)
        plt.show()

    @staticmethod
    def draw_correlation(input_array, output_array, category_id_list, category_names, feature_names):
        category_id_set = set(category_id_list)
        category_id_array = np.array(category_id_list)
        for i_feature in range(input_array.shape[1]):
            plt.figure()
            plt.title(feature_names[i_feature])
            plot_list, plot_names = [], []
            i_category = 0
            for category_id in category_id_set:
                category_name = category_names[category_id]
                plot_names.append(category_name)
                category_index = np.where(category_id_array == category_id)[0]
                category_plot, = plt.plot(input_array[category_index, i_feature], output_array[category_index],
                                          '.', color=COLORS[i_category])
                plot_list.append(category_plot)
                i_category += 1
            plt.legend(plot_list, plot_names)

    @staticmethod
    def clean_all_data(all_sub_data_struct):
        i_step = 0
        input_list, output_list = all_sub_data_struct.get_input_output_list()
        while i_step < len(all_sub_data_struct):
            # delete steps without a valid loading rate
            strikes = np.where(input_list[i_step][:, 6] == 1)[0]
            if np.max(output_list[i_step]) <= 0:
                all_sub_data_struct.pop(i_step)
            # delete steps without a valid strike time
            elif len(strikes) != 1:
                all_sub_data_struct.pop(i_step)
            else:
                # step number only increase when no pop happens
                i_step += 1
        return all_sub_data_struct

    @staticmethod
    def convert_output(output_all_list):
        step_num = len(output_all_list)
        step_output = np.zeros([step_num])
        for i_step in range(step_num):
            step_output[i_step] = np.max(output_all_list[i_step])
        return step_output

    # convert the input from list to ndarray
    @staticmethod
    def convert_input_0(input_all_list, sensor_sampling_fre, start_phase_0=0, end_phase_0=0.12,
                        start_phase_1=0.62, end_phase_1=0.75,
                        start_phase_2=0, end_phase_2=1):
        """
        Min, max, sum feature based solution
        """
        step_num = len(input_all_list)
        step_input = np.zeros([step_num, 4])
        for i_step in range(step_num):
            acc_data = input_all_list[i_step][:, 0:3]
            gyr_data = input_all_list[i_step][:, 3:6]
            strike_data = input_all_list[i_step][:, 6]
            step_len = input_all_list[i_step].shape[0]
            acc_normed = np.linalg.norm(acc_data, axis=1)

            # feature 0, strike time
            strike_sample_num = np.where(strike_data == 1)[0][0]
            strike_sample_num_percent = strike_sample_num / step_len
            step_input[i_step, 0] = strike_sample_num_percent

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
            step_input[i_step, 2] = np.max(acc_slice_fit)

            # feature 4, step length
            step_input[i_step, 3] = step_len

        feature_names = ['strike_sample_num_percent', 'rotation after strike', 'max acc', 'step length']
        return step_input, feature_names

    # convert the input from list to ndarray
    @staticmethod
    def convert_input_1(input_all_list, sampling_fre):
        """
        CNN based algorithm
        """
        step_num = len(input_all_list)
        sample_before_strike = int(8 * (sampling_fre / 100))
        sample_after_strike = int(8 * (sampling_fre / 100))
        win_len = sample_after_strike + sample_before_strike        # convolution kernel length
        step_input = np.zeros([step_num, win_len, 2])
        for i_step in range(step_num):
            acc_data = input_all_list[i_step][:, 2:3]
            gyr_data = input_all_list[i_step][:, 3:4]
            strike_data = input_all_list[i_step][:, 6]
            strike_sample_num = np.where(strike_data == 1)[0][0]
            start_sample = strike_sample_num - sample_before_strike
            end_sample = strike_sample_num + sample_after_strike
            step_data = np.column_stack([acc_data[start_sample:end_sample, :], gyr_data[start_sample:end_sample, :]])
            step_input[i_step, :, :] = step_data
        return step_input

    def linear_regression_solution(self):
        model = LinearRegression()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    def nn_solution(self):
        model = MLPRegressor(hidden_layer_sizes=40, activation='logistic')
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    def cnn_solution(self):
        model = Sequential()
        # debug kernel_size = 1, 看下下一层有几个输入
        # deliberately set kernel_size equal to input_shape[0] so that
        input_shape = self._x_train.shape
        model.add(Conv1D(filters=10, kernel_size=input_shape[1], input_shape=input_shape[1:]))
        model.add(Flatten())
        model.add(Dense(80, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_nn(model, 'loading rate')
        plt.show()

    def cnn_channel_independent_solution(self):
        input_shape = self._x_train.shape
        feature_outputs = []
        inputs = Input((input_shape[1:]))
        multiple_features = Lambda(ProcessorLR.splitter)(inputs)
        # for each feature
        for feature in multiple_features:
            # feature_output = Conv1D(filters=100, kernel_size=input_shape[1])(feature)
            feature_output = Conv1D(filters=5, kernel_size=int(input_shape[1]/2))(feature)
            feature_outputs.append(feature_output)
        joined_outputs = concatenate(feature_outputs)
        joined_outputs = Activation('relu')(joined_outputs)
        outputs = Flatten()(joined_outputs)
        outputs = Dense(80, activation='relu')(outputs)
        outputs = Dense(20, activation='relu')(outputs)
        outputs = Dense(1, activation='relu')(outputs)
        model = Model(inputs, outputs)
        plot_model(model, to_file='model.png')
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_nn(model, 'loading rate')
        plt.show()

    #function to split the input in multiple outputs
    @staticmethod
    def splitter(x):
        feature_num = x.shape[2]
        return [x[:, :, i:i+1] for i in range(feature_num)]













