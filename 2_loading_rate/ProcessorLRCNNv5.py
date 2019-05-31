"""
cross validation
(1) No resampling
(2) Reduced network size
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.layers import *
from ProcessorLRCNNv3 import ProcessorLRCNNv3
from ProcessorLR import ProcessorLR
from keras.models import Sequential, Model
from AllSubData import AllSubData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from const import TRIAL_NAMES, SUB_NAMES


class ProcessorLRCNNv5(ProcessorLRCNNv3):
    def __init__(self, sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=True, do_input_norm=True,
                 do_output_norm=False):
        super().__init__(sub_and_trials, None, sensor_sampling_fre, strike_off_from_IMU,
                         split_train=False, do_input_norm=do_input_norm)
        self.do_output_norm = do_output_norm

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
        aux_input = ProcessorLRCNNv3.clean_aux_input(aux_input)
        return step_input, aux_input

    def prepare_data_cross_vali(self, test_set_sub_num=1):
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list)
        input_list, output_list = train_all_data_list.get_input_output_list()
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()

        sub_id_set_tuple = tuple(set(sub_id_list))
        sample_num = len(input_list)
        sub_num = len(self.train_sub_and_trials.keys())
        folder_num = int(np.ceil(sub_num / test_set_sub_num))        # the number of cross validation times
        predict_result_all = np.zeros([0, 1])
        for i_folder in range(folder_num):
            test_id_list = sub_id_set_tuple[test_set_sub_num*i_folder:test_set_sub_num*(i_folder+1)]
            print('\ntest subjects: ')
            for test_id in test_id_list:
                print(SUB_NAMES[test_id])
            input_list_train, input_list_test, output_list_train, output_list_test = [], [], [], []
            for i_sample in range(sample_num):
                if sub_id_list[i_sample] in test_id_list:
                    input_list_test.append(input_list[i_sample])
                    output_list_test.append(output_list[i_sample])
                else:
                    input_list_train.append(input_list[i_sample])
                    output_list_train.append(output_list[i_sample])

            self._x_train, self._x_train_aux = self.convert_input(input_list_train, self.sensor_sampling_fre)
            self._y_train = ProcessorLR.convert_output(output_list_train)
            self._x_test, self._x_test_aux = self.convert_input(input_list_test, self.sensor_sampling_fre)
            self._y_test = ProcessorLR.convert_output(output_list_test)

            if self.do_output_norm:
                self._y_train = self.norm_output(self._y_train, self._x_train_aux[:, 0])
                self._x_test_aux_ori = self._x_test_aux.copy()

            # do input normalization
            if self.do_input_norm:
                self.norm_input()

            y_pred = self.cnn_solution().reshape([-1, 1])
            predict_result_all = np.row_stack([predict_result_all, y_pred])
        predict_result_all = predict_result_all.ravel()
        y_true = ProcessorLR.convert_output(output_list)
        Evaluation.plot_nn_result_cate_color(y_true, predict_result_all, trial_id_list, TRIAL_NAMES, 'loading rate')
        plt.show()

    def cnn_solution(self):
        main_input_shape = self._x_train.shape
        main_input = Input((main_input_shape[1:]), name='main_input')
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv1D(filters=5, kernel_size=20)(main_input)
        tower_1 = MaxPool1D(pool_size=11)(tower_1)

        # for each feature, add 10 * 1 cov kernel
        tower_2 = Conv1D(filters=5, kernel_size=9)(main_input)
        tower_2 = MaxPool1D(pool_size=3, strides=3)(tower_2)
        tower_2 = Conv1D(filters=5, kernel_size=4)(tower_2)

        # for each feature, add 5 * 1 cov kernel
        tower_3 = Conv1D(filters=5, kernel_size=4)(main_input)
        tower_3 = MaxPool1D(pool_size=3, strides=3)(tower_3)
        tower_3 = Conv1D(filters=5, kernel_size=4)(tower_3)
        tower_3 = MaxPool1D(pool_size=6)(tower_3)

        joined_outputs = concatenate([tower_1, tower_2, tower_3], axis=1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])
        aux_joined_outputs = Dense(30, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(15, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        # model.reset_states()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        model.save_weights('model_sub_' + str('sub'))
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred, self._x_test_aux_ori[:, 0])
        return y_pred

    def norm_output(self, output, step_len):
        step_num = output.shape[0]
        for i_step in range(step_num):
            output[i_step] = output[i_step] / step_len[i_step]
        self.standard_scalar = MinMaxScaler()
        output = output.reshape(-1, 1)
        output = self.standard_scalar.fit_transform(output)
        return output.reshape(-1,)

    def norm_output_reverse(self, output, step_len):
        output = output.reshape(-1, 1)
        output = self.standard_scalar.inverse_transform(output)
        step_num = output.shape[0]
        for i_step in range(step_num):
            output[i_step] = output[i_step] * step_len[i_step]
        return output.reshape(-1,)

