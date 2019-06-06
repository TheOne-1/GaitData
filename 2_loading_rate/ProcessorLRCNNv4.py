"""
cross validation
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.layers import *
from ProcessorLRCNNv3 import ProcessorLRCNNv3
from ProcessorLR import ProcessorLR
from keras.models import Model

from const import TRIAL_NAMES


class ProcessorLRCNNv4(ProcessorLRCNNv3):
    def __init__(self, sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=True, do_input_norm=True):
        super().__init__(sub_and_trials, None, sensor_sampling_fre, strike_off_from_IMU,
                         split_train=False, do_input_norm=do_input_norm)

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
        tower_2 = Conv1D(filters=5, kernel_size=10)(main_input)
        tower_2 = MaxPool1D(pool_size=21)(tower_2)

        # for each feature, add 5 * 1 cov kernel
        tower_3 = Conv1D(filters=5, kernel_size=5)(main_input)
        tower_3 = MaxPool1D(pool_size=26)(tower_3)

        joined_outputs = concatenate([tower_1, tower_2, tower_3], axis=1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(40, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='relu')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(model)
        return y_pred
