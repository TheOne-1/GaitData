"""
Conv template, improvements:
(1) add input such as subject height, step length, strike occurance time
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.layers import *
from ProcessorLR import ProcessorLR
from keras.models import Model
from sklearn.model_selection import train_test_split
from const import TRIAL_NAMES
from convert_model import convert
from scipy.stats import pearsonr
import json


class ProcessorLRCNNv3_1(ProcessorLR):
    def prepare_data(self):
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        self._x_train, self._x_train_aux = self.convert_input(input_list, self.sensor_sampling_fre)
        self._y_train = ProcessorLR.convert_output(output_list)

        # # !!! debugging
        # for step_data in input_list:
        #     plt.plot(step_data[:, 2])
        # plt.show()

        if not self.split_train:
            test_all_data_list = ProcessorLR.clean_all_data(self.test_all_data_list, self.sensor_sampling_fre)
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

        if self.do_output_norm:
            self.norm_output()

    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        resample_len = self.sensor_sampling_fre
        data_clip_start, data_clip_end = int(resample_len * 0.5), int(resample_len * 0.75)
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

        # # !!! debugging
        # plt.figure()
        # for i_step in range(step_num):
        #     plt.plot(step_input[i_step, :, 2])
        # plt.grid()
        # plt.show()

        aux_input = ProcessorLRCNNv3_1.clean_aux_input(aux_input)
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

    # @staticmethod
    # def build_model():

    def cnn_solution(self):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.01)
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv1D(filters=11, kernel_size=15*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPool1D(pool_size=10*base_size+1)(tower_1)

        # for each feature, add 5 * 1 cov kernel
        tower_3 = Conv1D(filters=11, kernel_size=5*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPool1D(pool_size=20*base_size+1)(tower_3)

        # for each feature, add 5 * 1 cov kernel
        tower_4 = Conv1D(filters=11, kernel_size=2*base_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPool1D(pool_size=23*base_size+1)(tower_4)

        # for each feature, add 5 * 1 cov kernel
        tower_5 = Conv1D(filters=11, kernel_size=1, kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPool1D(pool_size=50)(tower_5)

        joined_outputs = concatenate([tower_1, tower_3, tower_4, tower_5], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(20, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(15, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(10, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)

        if self.split_train:
            my_evaluator.plot_nn_result(self._y_test, y_pred, 'loading rate')
        else:
            my_evaluator.plot_nn_result_cate_color(self._y_test, y_pred, self.test_trial_id_list, TRIAL_NAMES,
                                                   'loading rate')
        self.model = model

    def save_model_and_param(self):
        self.model.save('lr_model.h5', include_optimizer=False)
        convert('lr_model.h5', 'fdeep_model.json')

        scalar_param = {'main_max_vals': self.main_max_vals, 'main_min_vals': self.main_min_vals,
                        'aux_max_vals': self.aux_max_vals, 'aux_min_vals': self.aux_min_vals,
                        'result_scale': self.output_minmax_scalar.scale_[0],
                        'result_min': self.output_minmax_scalar.min_[0]}

        scalar_file = 'scalar_param.json'
        with open(scalar_file, 'w') as param_file:
            print(json.dumps(scalar_param, sort_keys=True, indent=4, separators=(',', ': ')), file=param_file)

        plt.show()

    def to_generate_figure(self):
        y_pred = self.model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux}).ravel()
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)
        return self._y_test, y_pred


