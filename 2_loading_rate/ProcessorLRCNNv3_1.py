"""
Conv template, improvements:
(1) add input such as subject height, step length, strike occurance time
(2) reduced cov kernel size
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.layers import *
from ProcessorLR import ProcessorLR
from keras.models import Sequential, Model
from AllSubData import AllSubData
from sklearn.model_selection import train_test_split
from const import RUNNING_TRIALS, TRIAL_NAMES


class ProcessorLRCNNv3_1(ProcessorLR):
    def prepare_data(self):
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list)
        input_list, output_list = train_all_data_list.get_input_output_list()
        self._x_train, self._x_train_aux = self.convert_input(input_list, self.sensor_sampling_fre)
        self._y_train = ProcessorLR.convert_output(output_list)

        if not self.split_train:
            test_all_data_list = ProcessorLR.clean_all_data(self.test_all_data_list)
            input_list, output_list = test_all_data_list.get_input_output_list()
            self.test_sub_id_list = test_all_data_list.get_sub_id_list()
            self.test_trial_id_list = test_all_data_list.get_trial_id_list()
            self._x_test, self._x_test_aux = self.convert_input(input_list, self.sensor_sampling_fre)
            self._y_test = ProcessorLR.convert_output(output_list)
        else:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._x_train_aux, self._x_test_aux, self._y_train, self._y_test =\
                train_test_split(self._x_train, self._x_train_aux, self._y_train, test_size=0.33)

        # do input normalization
        if self.do_input_norm:
            self.norm_input()

    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        resample_len = 100
        data_clip_start, data_clip_end = 50, 80
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, 6])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:6]
            for i_channel in range(6):
                channel_resampled = ProcessorLR.resample_channel(acc_gyr_data[:, i_channel], resample_len)
                step_input[i_step, :, i_channel] = channel_resampled[data_clip_start:data_clip_end]
                step_len = acc_gyr_data.shape[0]
                aux_input[i_step, 0] = step_len
                strike_sample_num = np.where(input_all_list[i_step][:, 6] == 1)[0]
                aux_input[i_step, 1] = strike_sample_num
        aux_input = ProcessorLRCNNv3_1.clean_aux_input(aux_input)

        return step_input, aux_input

    @staticmethod
    def clean_aux_input(aux_input):
        """
        replace zeros by the average
        :param aux_input:
        :return:
        """
        aux_input_median = np.median(aux_input, axis=0)
        for i_channel in range(aux_input.shape[1]):
            zero_indexes = np.where(aux_input[:, i_channel] == 0)[0]
            aux_input[zero_indexes, i_channel] = aux_input_median[i_channel]
            if len(zero_indexes) != 0:
                print('Zero encountered in aux input. Replaced by the median')
        return aux_input

    def cnn_solution(self):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv1D(filters=10, kernel_size=20)(main_input)
        tower_1 = MaxPool1D(pool_size=11)(tower_1)

        # # for each feature, add 10 * 1 cov kernel
        # tower_2 = Conv1D(filters=5, kernel_size=10)(main_input)
        # tower_2 = MaxPool1D(pool_size=21)(tower_2)
        #
        # # for each feature, add 5 * 1 cov kernel
        # tower_3 = Conv1D(filters=5, kernel_size=5)(main_input)
        # tower_3 = MaxPool1D(pool_size=26)(tower_3)

        # joined_outputs = concatenate([tower_1, tower_2, tower_3], axis=1)
        joined_outputs = Activation('relu')(tower_1)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(20, activation='sigmoid')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        model.load_weights('model_weights_all_sub.h5')
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        self.check_pred_zero(y_pred, self._x_test)
        if self.split_train:
            my_evaluator.plot_nn_result(self._y_test, y_pred, 'loading rate')
        else:
            my_evaluator.plot_nn_result_cate_color(self._y_test, y_pred, self.test_trial_id_list, TRIAL_NAMES, 'loading rate')
        # model.save_weights('model_weights_all_sub.h5')
        plt.show()

    def check_pred_zero(self, y_pred, x_test):
        """
        To check why there are zeros in predicted values
        :return:
        """
        zero_indexes = np.where(y_pred < 1e-3)[0]
        non_zero_indexes = np.where(y_pred > 1e-3)[0]

        x=1
