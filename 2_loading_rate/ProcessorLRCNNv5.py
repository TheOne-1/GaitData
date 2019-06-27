"""
cross validation
(1) No resampling
(2) Reduced network size
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.layers import *
from ProcessorLRCNNv3_1 import ProcessorLRCNNv3_1
from ProcessorLRCNNv3_2 import ProcessorLRCNNv3_2
from ProcessorLR import ProcessorLR
from keras.models import Model
import pandas as pd
from const import SUB_NAMES


class ProcessorLRCNNv5(ProcessorLRCNNv3_1):
    def __init__(self, sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=True, do_input_norm=True, do_output_norm=True):
        super().__init__(sub_and_trials, None, sensor_sampling_fre, strike_off_from_IMU,
                         split_train=False, do_input_norm=do_input_norm, do_output_norm=do_output_norm)

    def prepare_data_cross_vali(self, test_set_sub_num=1):
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        sub_id_list = train_all_data_list.get_sub_id_list()
        trial_id_list = train_all_data_list.get_trial_id_list()

        sub_id_set_tuple = tuple(set(sub_id_list))
        sample_num = len(input_list)
        sub_num = len(self.train_sub_and_trials.keys())
        folder_num = int(np.ceil(sub_num / test_set_sub_num))        # the number of cross validation times
        predict_result_df = pd.DataFrame()
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

            # do input normalization
            if self.do_input_norm:
                self.norm_input()

            if self.do_output_norm:
                self.norm_output()

            y_pred = self.cnn_solution().reshape([-1, 1])
            if self.do_output_norm:
                y_pred = self.norm_output_reverse(y_pred)
            pearson_coeff, RMSE, mean_error = Evaluation.plot_nn_result(self._y_test, y_pred, title=SUB_NAMES[test_id_list[0]])

            predict_result_df = Evaluation.insert_prediction_result(
                predict_result_df, SUB_NAMES[test_id_list[0]], pearson_coeff, RMSE, mean_error)
        Evaluation.export_prediction_result(predict_result_df)
        plt.show()

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
        return y_pred













