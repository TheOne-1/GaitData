
from AllSubDataGRF import AllSubDataGRF
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.layers import *
from ProcessorLR import ProcessorLR
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from const import TRIAL_NAMES


class ProcessorGRF(ProcessorLR):
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=False,
                 split_train=False, do_input_norm=True):
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.sensor_sampling_fre = sensor_sampling_fre
        self.strike_off_from_IMU = strike_off_from_IMU
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.param_name = 'LR'
        train_all_data = AllSubDataGRF(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
        self.train_all_data_list = train_all_data.get_all_data()
        if test_sub_and_trials is not None:
            test_all_data = AllSubDataGRF(self.test_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
            self.test_all_data_list = test_all_data.get_all_data()

    def prepare_data(self):
        input_list, output_list = self.train_all_data_list.get_input_output_list()
        self._x_train, self._x_train_aux = self.convert_input(input_list, self.sensor_sampling_fre)
        self._y_train = self.convert_output(output_list)

        if not self.split_train:
            input_list, output_list = self.test_all_data_list.get_input_output_list()
            self.test_sub_id_list = self.test_all_data_list.get_sub_id_list()
            self.test_trial_id_list = self.test_all_data_list.get_trial_id_list()
            self._x_test, self._x_test_aux = self.convert_input(input_list, self.sensor_sampling_fre)
            self._y_test = self.convert_output(output_list)
        else:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._x_train_aux, self._x_test_aux, self._y_train, self._y_test =\
                train_test_split(self._x_train, self._x_train_aux, self._y_train, test_size=0.33)

        # do input normalization
        if self.do_input_norm:
            self.norm_input()

    def convert_input(self, input_all_list, sensor_sampling_fre):
        step_num = len(input_all_list)
        resample_len = int(0.4 * self.sensor_sampling_fre)
        step_input = np.zeros([step_num, resample_len, 6])
        aux_input = np.zeros([step_num, 1])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:6]
            for i_channel in range(6):
                channel_resampled = ProcessorLR.resample_channel(acc_gyr_data[:, i_channel], resample_len)
                step_input[i_step, :, i_channel] = channel_resampled
                step_len = acc_gyr_data.shape[0]
                aux_input[i_step, 0] = step_len
        return step_input, aux_input

    def convert_output(self, output_all_list):
        step_num = len(output_all_list)
        resample_len = int(0.3 * self.sensor_sampling_fre)
        step_output = np.zeros([step_num, resample_len])
        for i_step in range(step_num):
            output_resampled = ProcessorLR.resample_channel(output_all_list[i_step], resample_len)
            step_output[i_step, :] = output_resampled
        return step_output

    def cnn_solution(self):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.sensor_sampling_fre*0.01)
        full_size = main_input_shape[1]

        # for each feature, add 20 * 1 cov kernel
        kernel_size = full_size
        tower_1 = Conv1D(filters=9, kernel_size=kernel_size)(main_input)
        tower_1 = MaxPool1D(pool_size=full_size + 1 - kernel_size)(tower_1)

        # for each feature, add 5 * 1 cov kernel
        kernel_size = 10*base_size
        tower_3 = Conv1D(filters=9, kernel_size=kernel_size)(main_input)
        tower_3 = MaxPool1D(pool_size=full_size + 1 - kernel_size)(tower_3)

        # for each feature, add 5 * 1 cov kernel
        kernel_size = 3*base_size
        tower_4 = Conv1D(filters=9, kernel_size=kernel_size)(main_input)
        tower_4 = MaxPool1D(pool_size=full_size + 1 - kernel_size)(tower_4)

        # for each feature, add 5 * 1 cov kernel
        kernel_size = 1*base_size
        tower_5 = Conv1D(filters=9, kernel_size=kernel_size)(main_input)
        tower_5 = MaxPool1D(pool_size=full_size + 1 - kernel_size)(tower_5)

        joined_outputs = concatenate([tower_1, tower_3, tower_4, tower_5], axis=2)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(1,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(50*base_size, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50*base_size, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50*base_size, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(30*base_size, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        my_evaluator.plot_continuous_result(self._y_test, y_pred, 'GRFz')
        plt.show()

