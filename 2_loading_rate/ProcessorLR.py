from Evaluation import Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from AllSubData import AllSubData
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import plot_model
import scipy.interpolate as interpo
from sklearn.ensemble import GradientBoostingRegressor
from const import SUB_NAMES, TRIAL_NAMES, COLORS, DATA_COLUMNS_XSENS
import scipy.stats as stats
from sklearn.model_selection import train_test_split


class ProcessorLR:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=False,
                 split_train=False):
        param_name = 'LR'
        train_all_data = AllSubData(train_sub_and_trials, param_name, sensor_sampling_fre, strike_off_from_IMU)
        train_all_data_list = train_all_data.get_all_data()
        train_all_data_list = ProcessorLR.clean_all_data(train_all_data_list)
        input_list, output_list = train_all_data_list.get_input_output_list()
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()
        self._x_train, feature_names = ProcessorLR.convert_input_2(input_list, sensor_sampling_fre)
        self._y_train = ProcessorLR.convert_output(output_list)
        # ProcessorLR.gait_phase_and_correlation(input_list, self._y_train, channels=range(6))
        # ProcessorLR.draw_correlation(self._x_train, self._y_train, sub_id_list, SUB_NAMES, feature_names)
        # ProcessorLR.draw_correlation(self._x_train, self._y_train, trial_id_list, TRIAL_NAMES, feature_names)
        # plt.show()

        if not split_train:
            test_all_data = AllSubData(test_sub_and_trials, param_name, sensor_sampling_fre, strike_off_from_IMU)
            test_all_data_list = test_all_data.get_all_data()
            test_all_data_list = ProcessorLR.clean_all_data(test_all_data_list)
            input_list, output_list = test_all_data_list.get_input_output_list()
            self._x_test, feature_names = ProcessorLR.convert_input_1(input_list, sensor_sampling_fre)
            self._y_test = ProcessorLR.convert_output(output_list)
        else:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
                self._x_train, self._y_train, test_size=0.33)

    @staticmethod
    def gait_phase_and_correlation(input_list, output_array, channels=range(6)):
        sample_num = len(input_list)
        resample_len = 100
        plt.figure()
        plot_list = []
        for i_channel in channels:
            input_array = np.zeros([sample_num, resample_len])
            for i_sample in range(sample_num):
                channel_data = input_list[i_sample][:, i_channel]
                channel_data_resampled = ProcessorLR.resample_channel(channel_data, resample_len)
                input_array[i_sample, :] = channel_data_resampled
            pear_correlations = np.zeros([resample_len])
            for phase in range(resample_len):
                pear_correlations[phase] = stats.pearsonr(input_array[:, phase], output_array)[0]
            channel_plot, = plt.plot(pear_correlations, color=COLORS[i_channel])
            plot_list.append(channel_plot)
        plt.xlabel('gait phase')
        plt.ylabel('correlation')
        plt.legend(plot_list, DATA_COLUMNS_XSENS[0:max(channels)+1])
        plt.grid()
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
                if 'mini' in category_name:
                    plot_pattern = 'x'
                else:
                    plot_pattern = '.'
                plot_names.append(category_name)
                category_index = np.where(category_id_array == category_id)[0]
                category_plot, = plt.plot(input_array[category_index, i_feature], output_array[category_index],
                                          plot_pattern, color=COLORS[i_category])
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
    def convert_input_0(input_all_list, sensor_sampling_fre):
        """
        Min, max, sum feature based solution
        """
        step_num = len(input_all_list)
        step_input = np.zeros([step_num, 8])
        for i_step in range(step_num):
            # acc_data = input_all_list[i_step][:, 0:3]
            # gyr_data = input_all_list[i_step][:, 3:6]
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

            # feature 4, gyr_y 53 - 57
            start_phase_4, end_phase_4 = 53, 57
            step_input[i_step, 4] = np.mean(acc_gyr_data_resampled[start_phase_4:end_phase_4, 4])

            # feature 5, gyr_y 64 - 66
            start_phase_5, end_phase_5 = 64, 66
            step_input[i_step, 5] = np.mean(acc_gyr_data_resampled[start_phase_5:end_phase_5, 4])

            # feature 6, gyr_z 53 - 57
            start_phase_6, end_phase_6 = 42, 48
            step_input[i_step, 6] = np.mean(acc_gyr_data_resampled[start_phase_6:end_phase_6, 5])

            # feature 7, gyr_x 50 - 60
            start_phase_7, end_phase_7 = 50, 60
            step_input[i_step, 7] = np.mean(acc_gyr_data_resampled[start_phase_7:end_phase_7, 3])

        feature_names = ['strike_sample_num_percent', 'step length', 'acc_z at heel strike', 'acc_z 52 - 55',
                         'gyr_y 53 - 57', 'gyr_y 64 - 66', 'gyr_z 53 - 57', 'gyr_x 50 - 60']
        return step_input, feature_names

    # convert the input from list to ndarray
    @staticmethod
    def convert_input_1(input_all_list, sensor_sampling_fre):
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

    # convert the input from list to ndarray
    @staticmethod
    def convert_input_2(input_all_list, sampling_fre):
        """
        CNN based algorithm
        """
        step_num = len(input_all_list)
        sample_before_strike = int(12 * (sampling_fre / 100))
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
        feature_names = None
        return step_input, feature_names

    # convert the input from list to ndarray
    @staticmethod
    def convert_input_3(input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        resample_len = 100
        step_input = np.zeros([step_num, resample_len, 6])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:6]
            for i_channel in range(6):
                step_input[i_step, :, i_channel] = ProcessorLR.resample_channel(acc_gyr_data[:, i_channel],
                                                                                resample_len)
            strike_data = input_all_list[i_step][:, 6]
            step_len = input_all_list[i_step].shape[0]

        feature_names = None
        return step_input, feature_names

    def linear_regression_solution(self):
        model = LinearRegression()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    def GBDT_solution(self):
        model = GradientBoostingRegressor()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        print(model.feature_importances_)
        plt.show()
        return model.feature_importances_

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

    def cnn_inception_solution(self):
        input_shape = self._x_train.shape
        inputs = Input((input_shape[1:]))
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv1D(filters=20, kernel_size=20)(inputs)
        tower_1 = MaxPool1D(pool_size=81)(tower_1)

        # for each feature, add 10 * 1 cov kernel
        tower_2 = Conv1D(filters=20, kernel_size=10)(inputs)
        tower_2 = MaxPool1D(pool_size=91)(tower_2)

        # for each feature, add 5 * 1 cov kernel
        tower_3 = Conv1D(filters=20, kernel_size=5)(inputs)
        tower_3 = MaxPool1D(pool_size=96)(tower_3)

        joined_outputs = concatenate([tower_1, tower_2, tower_3], axis=1)
        joined_outputs = Activation('relu')(joined_outputs)
        outputs = Flatten()(joined_outputs)
        outputs = Dense(40, activation='relu')(outputs)
        outputs = Dense(20, activation='relu')(outputs)
        outputs = Dense(1, activation='relu')(outputs)
        model = Model(inputs, outputs)
        plot_model(model, to_file='model.png')
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
        outputs = Dense(40, activation='relu')(outputs)
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

    @staticmethod
    def resample_channel(data_array, resampled_len):
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        data_len = data_array.shape[1]
        data_step = np.arange(0, data_len)
        resampled_step = np.linspace(0, data_len, resampled_len)
        tck, data_step = interpo.splprep(data_array, u=data_step, s=0)
        data_resampled = interpo.splev(resampled_step, tck, der=0)[0]
        return data_resampled









