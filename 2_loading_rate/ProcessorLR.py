from Evaluation import Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from AllSubData import AllSubData
from keras.layers import *
import scipy.interpolate as interpo
from sklearn.ensemble import GradientBoostingRegressor
from const import SUB_NAMES, TRIAL_NAMES, COLORS, DATA_COLUMNS_XSENS, DATA_COLUMNS_IMU
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json


class ProcessorLR:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=False,
                 split_train=False, do_input_norm=True, do_output_norm=False):
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.sensor_sampling_fre = sensor_sampling_fre
        self.strike_off_from_IMU = strike_off_from_IMU
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.do_output_norm = do_output_norm
        self.param_name = 'LR'
        train_all_data = AllSubData(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
        self.train_all_data_list = train_all_data.get_all_data()
        if test_sub_and_trials is not None:
            test_all_data = AllSubData(self.test_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
            self.test_all_data_list = test_all_data.get_all_data()

    def prepare_data(self):
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        self._x_train, feature_names = self.convert_input(input_list, self.sensor_sampling_fre)
        self._y_train = ProcessorLR.convert_output(output_list)

        if not self.split_train:
            test_all_data_list = ProcessorLR.clean_all_data(self.test_all_data_list, self.sensor_sampling_fre)
            input_list, output_list = test_all_data_list.get_input_output_list()
            self._x_test, feature_names = self.convert_input(input_list, self.sensor_sampling_fre)
            self._y_test = ProcessorLR.convert_output(output_list)
        else:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
                self._x_train, self._y_train, test_size=0.33)

        # do input normalization
        if self.do_input_norm:
            self.norm_input()

        if self.do_output_norm:
            self.norm_output()

    def find_feature(self):
        train_all_data = AllSubData(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
        train_all_data_list = train_all_data.get_all_data()
        train_all_data_list = ProcessorLR.clean_all_data(train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        x_train, feature_names = self.convert_input(input_list, self.sensor_sampling_fre)
        y_train = ProcessorLR.convert_output(output_list)
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()
        ProcessorLR.gait_phase_and_correlation(input_list, y_train, channels=range(6))
        ProcessorLR.draw_correlation(x_train, y_train, sub_id_list, SUB_NAMES, feature_names)
        ProcessorLR.draw_correlation(x_train, y_train, trial_id_list, TRIAL_NAMES, feature_names)
        plt.show()

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
    def clean_all_data(all_sub_data_struct, sensor_sampling_fre):
        i_step = 0
        input_list, output_list = all_sub_data_struct.get_input_output_list()
        min_time_between_strike_off = int(sensor_sampling_fre * 0.15)
        while i_step < len(all_sub_data_struct):
            # delete steps without a valid loading rate
            strikes = np.where(input_list[i_step][:, 6] == 1)[0]
            if np.max(output_list[i_step]) <= 0:
                all_sub_data_struct.pop(i_step)

            # delete steps without a valid strike time
            elif len(strikes) != 1:
                all_sub_data_struct.pop(i_step)

            # delete a step if the duration between strike and off is too short
            elif not min_time_between_strike_off < input_list[i_step].shape[0] - strikes[0]:
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
    def convert_input(self, input_all_list, sampling_fre):
        # this method has to be overwritten
        raise NotImplementedError('this convert_step_input method has to be overwritten')

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

    def norm_input(self):
        channel_num = self._x_train.shape[2]
        # save input scalar parameter
        self.main_max_vals,  self.main_min_vals = [], []
        for i_channel in range(channel_num):
            max_val = np.max(self._x_train[:, :, i_channel]) * 0.99
            min_val = np.min(self._x_train[:, :, i_channel]) * 0.99
            self._x_train[:, :, i_channel] = (self._x_train[:, :, i_channel] - min_val) / (max_val - min_val)
            self._x_test[:, :, i_channel] = (self._x_test[:, :, i_channel] - min_val) / (max_val - min_val)
            self.main_max_vals.append(max_val)
            self.main_min_vals.append(min_val)

        if hasattr(self, '_x_train_aux'):
            # MinMaxScaler is more suitable because StandardScalar will make the input greatly differ from each other
            aux_input_scalar = MinMaxScaler()
            self._x_train_aux = aux_input_scalar.fit_transform(self._x_train_aux)
            self._x_test_aux = aux_input_scalar.transform(self._x_test_aux)
            self.aux_max_vals = aux_input_scalar.data_max_.tolist()
            self.aux_min_vals = aux_input_scalar.data_min_.tolist()

    def norm_output(self):
        self.output_minmax_scalar = MinMaxScaler(feature_range=(1, 3))
        self._y_train = self._y_train.reshape(-1, 1)
        self._y_train = self.output_minmax_scalar.fit_transform(self._y_train)
        self.result_max_vals = self.output_minmax_scalar.data_max_[0]
        self.result_min_vals = self.output_minmax_scalar.data_min_[0]

    def norm_output_reverse(self, output):
        output = output.reshape(-1, 1)
        output = self.output_minmax_scalar.inverse_transform(output)
        return output.reshape(-1,)




