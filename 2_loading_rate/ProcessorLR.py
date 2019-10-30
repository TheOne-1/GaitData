"""
Conv template, improvements:
(1) add input such as subject height, step length, strike occurance time
"""
import matplotlib.pyplot as plt
from AllSubData import AllSubData
import scipy.interpolate as interpo
from const import SUB_NAMES, COLORS, DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE, SI_SR_TRIALS
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Evaluation import Evaluation
from keras import regularizers
from keras.layers import *
from keras.models import Model
import keras
from sklearn.model_selection import train_test_split
from const import TRIAL_NAMES
import numpy as np
import pandas as pd
import copy


class ProcessorLR:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, imu_locations, strike_off_from_IMU=1,
                 split_train=False, do_input_norm=True, do_output_norm=True):
        """

        :param train_sub_and_trials:
        :param test_sub_and_trials:
        :param imu_locations:
        :param strike_off_from_IMU: 0 for from plate, 1 for foot IMU, 2 for shank IMU
        :param split_train:
        :param do_input_norm:
        :param do_output_norm:
        """
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.imu_locations = imu_locations
        self.sensor_sampling_fre = MOCAP_SAMPLE_RATE
        self.strike_off_from_IMU = strike_off_from_IMU
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.do_output_norm = do_output_norm
        self.param_name = 'LR'
        train_all_data = AllSubData(self.train_sub_and_trials, imu_locations, self.param_name, self.sensor_sampling_fre,
                                    self.strike_off_from_IMU)
        self.train_all_data_list = train_all_data.get_all_data()
        if test_sub_and_trials is not None:
            test_all_data = AllSubData(self.test_sub_and_trials, imu_locations, self.param_name,
                                       self.sensor_sampling_fre, self.strike_off_from_IMU)
            self.test_all_data_list = test_all_data.get_all_data()

    def cnn_train_test(self):
        """
        The very basic condition, use the train set to train and use the test set to test.
        :return:
        """
        self.prepare_data()
        plt.show()
        self.do_normalization()
        self.define_cnn_model()
        self.evaluate_cnn_model()
        self.show_weights()
        keras.backend.clear_session()

        self.define_cnn_model()
        self.evaluate_cnn_model()
        self.show_weights()
        keras.backend.clear_session()

        self.define_cnn_model()
        self.evaluate_cnn_model()
        self.show_weights()
        keras.backend.clear_session()

        plt.show()

    def show_weights(self):
        """
        Show weights of the first Dense layer.
        :return:
        """
        for layer in self.model.layers:
            if isinstance(layer, Dense):
                weights = layer.get_weights()
                weights_sum = np.linalg.norm(weights[0], axis=1)
                plt.figure()
                plt.plot(weights_sum)
                break

    def cnn_cross_vali(self, test_date, test_name, test_set_sub_num=1, plot=True):
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        trial_ids = train_all_data_list.get_trial_id_list()
        self.channel_num = input_list[0].shape[1] - 1
        sub_id_list = train_all_data_list.get_sub_id_list()

        sub_id_set_tuple = tuple(set(sub_id_list))
        sample_num = len(input_list)
        sub_num = len(self.train_sub_and_trials.keys())
        folder_num = int(np.ceil(sub_num / test_set_sub_num))  # the number of cross validation times
        predict_result_df = pd.DataFrame()
        predicted_value_df = pd.DataFrame()  # save all the predicted values in case reviewer ask for more analysis
        for i_folder in range(folder_num):
            test_id_list = sub_id_set_tuple[test_set_sub_num * i_folder:test_set_sub_num * (i_folder + 1)]
            print('\ntest subjects: ')
            for test_id in test_id_list:
                print(SUB_NAMES[test_id])
            input_list_train, input_list_test, output_list_train, output_list_test, test_trial_ids = [], [], [], [], []
            for i_sample in range(sample_num):
                if sub_id_list[i_sample] in test_id_list:
                    input_list_test.append(input_list[i_sample])
                    output_list_test.append(output_list[i_sample])
                    test_trial_ids.append(trial_ids[i_sample])
                else:
                    input_list_train.append(input_list[i_sample])
                    output_list_train.append(output_list[i_sample])

            self._x_train, self._x_train_aux = self.convert_input(input_list_train, self.sensor_sampling_fre)
            self._y_train = ProcessorLR.convert_output(output_list_train)
            self._x_test, self._x_test_aux = self.convert_input(input_list_test, self.sensor_sampling_fre)
            self._y_test = ProcessorLR.convert_output(output_list_test)

            self.do_normalization()
            self.define_cnn_model()

            my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                      self._x_test_aux)
            y_pred = my_evaluator.evaluate_nn(self.model)
            if self.do_output_norm:
                y_pred = self.norm_output_reverse(y_pred)

            if plot:
                my_evaluator.plot_nn_result_cate_color(self._y_test, y_pred, test_trial_ids,
                                                       TRIAL_NAMES, title=SUB_NAMES[test_id_list[0]])
                plt.savefig('exported_figures/lr_' + SUB_NAMES[test_id_list[0]] + '.png')

            predict_result_df = self.save_detailed_results(predict_result_df, SUB_NAMES[test_id_list[0]],
                                                           self._y_test, y_pred, test_trial_ids)
            predicted_value_df = self.save_all_predicted_values(predicted_value_df, self._y_test, y_pred,
                                                                test_id_list[0], test_trial_ids)
        Evaluation.export_prediction_result(predict_result_df, test_date, test_name)
        Evaluation.export_predicted_values(predicted_value_df, test_date, test_name)

    @staticmethod
    def save_all_predicted_values(predicted_value_df, y_true, y_pred, sub_id, test_trial_ids):
        data_len = len(test_trial_ids)
        test_sub_ids_np = np.full([data_len], sub_id)
        test_trial_ids_np = np.array(test_trial_ids)
        current_df = pd.DataFrame(np.column_stack([test_sub_ids_np, test_trial_ids_np, y_true, y_pred]))
        predicted_value_df = predicted_value_df.append(current_df)
        return predicted_value_df

    @staticmethod
    def save_detailed_results(predict_result_df, sub_name, y_true, y_pred, test_trial_ids):
        """
        Record the result of each subject and each trial.
        :return:
        """
        test_trial_ids_np = np.array(test_trial_ids)
        pearson_coeff_list, RMSE_list, mean_error_list, abs_mean_error_list = [], [], [], []

        # result of each trial
        for trial_name in SI_SR_TRIALS:
            the_trial_id = TRIAL_NAMES.index(trial_name)
            y_true_trial = y_true[test_trial_ids_np == the_trial_id]
            y_pred_trial = y_pred[test_trial_ids_np == the_trial_id]
            pearson_coeff, RMSE, mean_error, abs_mean_error = Evaluation.get_all_scores(
                y_true_trial, y_pred_trial, precision=3)
            pearson_coeff_list.append(pearson_coeff)
            RMSE_list.append(RMSE)
            mean_error_list.append(mean_error)
            abs_mean_error_list.append(abs_mean_error)

        # overall result
        pearson_coeff, RMSE, mean_error, abs_mean_error = Evaluation.get_all_scores(
            y_true, y_pred, precision=3)
        pearson_coeff_list.append(pearson_coeff)
        RMSE_list.append(RMSE)
        mean_error_list.append(mean_error)
        abs_mean_error_list.append(abs_mean_error)

        sub_df_0 = pd.DataFrame([[sub_name, 'pearson correlation'] + pearson_coeff_list])
        sub_df_1 = pd.DataFrame([[sub_name, 'RMSE'] + RMSE_list])
        sub_df_2 = pd.DataFrame([[sub_name, 'mean error'] + mean_error_list])
        sub_df_3 = pd.DataFrame([[sub_name, 'absolute mean error'] + abs_mean_error_list])

        return pd.concat([predict_result_df, sub_df_0, sub_df_1, sub_df_2, sub_df_3])

    def prepare_data(self):
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        self.channel_num = input_list[0].shape[1] - 1
        self._x_train, self._x_train_aux = self.convert_input(input_list, self.sensor_sampling_fre)
        self._y_train = ProcessorLR.convert_output(output_list)

        if not self.split_train:
            test_all_data_list = ProcessorLR.clean_all_data(self.test_all_data_list, self.sensor_sampling_fre)
            input_list, output_list = test_all_data_list.get_input_output_list()
            self.test_sub_id_list = test_all_data_list.get_sub_id_list()
            self.test_trial_id_list = test_all_data_list.get_trial_id_list()
            self._x_test, self._x_test_aux = self.convert_input(input_list, self.sensor_sampling_fre)
            self._y_test = ProcessorLR.convert_output(output_list)
        else:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._x_train_aux, self._x_test_aux, self._y_train, self._y_test = \
                train_test_split(self._x_train, self._x_train_aux, self._y_train, test_size=0.33)

    def do_normalization(self):
        # do input normalization
        if self.do_input_norm:
            self.norm_input(feature_range=(-2, 2))

        if self.do_output_norm:
            self.norm_output()

        # reshape the input to fit tensorflow
        main_input_shape = list(self._x_train.shape)
        self._x_train = self._x_train.reshape(main_input_shape[0], main_input_shape[1], main_input_shape[2], 1)
        self._x_test = self._x_test.reshape(self._x_test.shape[0], main_input_shape[1], main_input_shape[2], 1)

    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre, data_clip_start=-12, data_clip_end=30):
        """
        input start from strike-20 to strike+20
        """
        step_num = len(input_all_list)
        # data_clip_start, data_clip_end = -40, 25
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, self.channel_num])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:self.channel_num]
            step_len = acc_gyr_data.shape[0]
            aux_input[i_step, 0] = step_len
            strike_sample_num = int(np.where(input_all_list[i_step][:, -1] == 1)[0])
            aux_input[i_step, 1] = strike_sample_num
            step_input[i_step, :, :] = acc_gyr_data[
                                       strike_sample_num + data_clip_start:strike_sample_num + data_clip_end, :]

        aux_input = ProcessorLR.clean_aux_input(aux_input)
        return step_input, aux_input

    @staticmethod
    def clean_aux_input(aux_input):
        """
        replace zeros by the average
        :param aux_input:
        :return:
        """
        # replace zeros
        aux_input_median = np.median(aux_input, axis=0)
        for i_channel in range(aux_input.shape[1]):
            zero_indexes = np.where(aux_input[:, i_channel] == 0)[0]
            aux_input[zero_indexes, i_channel] = aux_input_median[i_channel]
            if len(zero_indexes) != 0:
                print('Zero encountered in aux input. Replaced by the median')
        return aux_input

    def define_cnn_model(self):
        """
        Convolution kernel shape changed from 1D to 2D.
        :return:
        """
        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        # base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.001)

        kernel_size = np.array([3, main_input_shape[2]])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_2 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPooling2D(pool_size=pool_size)(tower_2)

        kernel_size = np.array([10, 1])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_3 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPooling2D(pool_size=pool_size)(tower_3)

        kernel_size = np.array([3, 1])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_4 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPooling2D(pool_size=pool_size)(tower_4)

        # for each feature, add 20 * 1 cov kernel
        kernel_size = np.array([1, main_input_shape[2]])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_5 = Conv2D(filters=20, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPooling2D(pool_size=pool_size)(tower_5)

        joined_outputs = concatenate([tower_2, tower_3, tower_4, tower_5], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(10, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model

    def evaluate_cnn_model(self):
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(self.model)
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)

        if self.split_train:
            my_evaluator.plot_nn_result(self._y_test, y_pred, 'loading rate')
        else:
            my_evaluator.plot_nn_result_cate_color(self._y_test, y_pred, self.test_trial_id_list, TRIAL_NAMES,
                                                   'loading rate')
        return y_pred

    def save_cnn_model(self, model_name='lr_model'):
        self.model.save(model_name + '.h5', include_optimizer=False)

    def to_generate_figure(self):
        y_pred = self.model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux}).ravel()
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)
        return self._y_test, y_pred

    def find_feature(self):
        train_all_data = AllSubData(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre,
                                    self.strike_off_from_IMU)
        train_all_data_list = train_all_data.get_all_data()
        train_all_data_list = ProcessorLR.clean_all_data(train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        x_train, feature_names = self.convert_input(input_list, self.sensor_sampling_fre)
        y_train = ProcessorLR.convert_output(output_list)
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()
        ProcessorLR.gait_phase_and_correlation(input_list, y_train, channels=range(self.channel_num))
        ProcessorLR.draw_correlation(x_train, y_train, sub_id_list, SUB_NAMES, feature_names)
        ProcessorLR.draw_correlation(x_train, y_train, trial_id_list, TRIAL_NAMES, feature_names)
        plt.show()

    @staticmethod
    def gait_phase_and_correlation(input_list, output_array, channels):
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
        plt.legend(plot_list, DATA_COLUMNS_XSENS[0:max(channels) + 1])
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
            strikes = np.where(input_list[i_step][:, -1] == 1)[0]
            if np.max(output_list[i_step]) <= 0:
                all_sub_data_struct.pop(i_step)

            # delete steps without a valid strike time
            elif len(strikes) != 1:
                all_sub_data_struct.pop(i_step)

            # delete a step if the duration between strike and off is too short
            # this can ensure the sample number after strike is larger than 30
            elif not min_time_between_strike_off < input_list[i_step].shape[0] - strikes[0]:
                all_sub_data_struct.pop(i_step)

            elif np.isnan(input_list[i_step]).any():
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

    def norm_input(self, feature_range=(0, 1)):
        channel_num = self._x_train.shape[2]
        # save input scalar parameter
        self.main_max_vals,  self.main_min_vals = [], []
        for i_channel in range(channel_num):
            max_val = np.max(self._x_train[:, :, i_channel])
            min_val = np.min(self._x_train[:, :, i_channel])
            self._x_train[:, :, i_channel] = (self._x_train[:, :, i_channel] - min_val) / (max_val - min_val) * (feature_range[1] - feature_range[0]) + feature_range[0]
            self._x_test[:, :, i_channel] = (self._x_test[:, :, i_channel] - min_val) / (max_val - min_val) * (feature_range[1] - feature_range[0]) + feature_range[0]

        if hasattr(self, '_x_train_aux'):
            # MinMaxScaler is more suitable because StandardScalar will make the input greatly differ from each other
            aux_input_scalar = MinMaxScaler()
            self._x_train_aux = aux_input_scalar.fit_transform(self._x_train_aux)
            self._x_test_aux = aux_input_scalar.transform(self._x_test_aux)

    def norm_output(self):
        self.output_minmax_scalar = MinMaxScaler(feature_range=(1, 3))
        self._y_train = self._y_train.reshape(-1, 1)
        self._y_train = self.output_minmax_scalar.fit_transform(self._y_train)
        self.result_max_vals = self.output_minmax_scalar.data_max_[0]
        self.result_min_vals = self.output_minmax_scalar.data_min_[0]

    def norm_output_reverse(self, output):
        output = output.reshape(-1, 1)
        output = self.output_minmax_scalar.inverse_transform(output)
        return output.reshape(-1, )
