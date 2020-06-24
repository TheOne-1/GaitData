"""To answer the question: "It would be helpful to include the amount of data available for training models
 from each subject and information about the relative success of model training for each IMU configuration."""

from ProcessorLR import ProcessorLR
from const import SUB_AND_SI_SR_TRIALS, TRIAL_NAMES, SUB_NAMES, SI_SR_TRIALS, RUNNING_TRIALS
import copy
import matplotlib.pyplot as plt
import keras
from Evaluation import Evaluation
import numpy as np
import pandas as pd
import os
from keras.layers import *
from keras import regularizers
from keras.models import Model


def create_folders(date):
    # create result folder
    result_main_folder = 'result_conclusion/' + date
    result_sub_folder_0 = 'result_conclusion/' + date + '/trial_summary/'
    result_sub_folder_1 = 'result_conclusion/' + date + '/step_result/'
    if not os.path.exists(result_main_folder):
        os.makedirs(result_main_folder)
    if not os.path.exists(result_sub_folder_0):
        os.makedirs(result_sub_folder_0)
    if not os.path.exists(result_sub_folder_1):
        os.makedirs(result_sub_folder_1)


class ProcessorLRModelParam(ProcessorLR):
    def __init__(self, train_sub_and_trials, test_sub_and_trials, imu_locations):
        super().__init__(train_sub_and_trials, test_sub_and_trials, imu_locations, strike_off_from_IMU=2,
                         split_train=False, do_input_norm=True, do_output_norm=True)
        self._hidden_layer_neurons = None

    def set_hidden_layer_neurons(self, hidden_layer_neurons):
        self._hidden_layer_neurons = hidden_layer_neurons

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

        """Define the shape of hidden layer based on requirement"""
        for layer_neuron in self._hidden_layer_neurons:
            aux_joined_outputs = Dense(layer_neuron, activation='relu')(aux_joined_outputs)

        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model

    def cnn_cross_vali_layer(self, test_date, test_name, layer_summary_np, test_set_sub_num=1, plot=False):
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
        predict_result_df = Evaluation.export_prediction_result(predict_result_df, test_date, test_name)
        Evaluation.export_predicted_values(predicted_value_df, test_date, test_name)

        all_sub_summary_df = predict_result_df[predict_result_df['subject name'] == 'absolute mean']
        layer_values = all_sub_summary_df['All trials'].values.reshape([-1])
        layer_summary_np = np.row_stack([layer_summary_np, layer_values])
        return layer_summary_np


def save_summary_np(summary_np, columns, segment_name_to_store, save_name):
    summary_df = pd.DataFrame(summary_np)
    summary_df.columns = columns
    summary_df.insert(0, 'layer_name', segment_name_to_store)

    file_path = 'result_conclusion/' + date + '/' + save_name + '.csv'
    i_file = 0
    while os.path.isfile(file_path):
        i_file += 1
        file_path = 'result_conclusion/' + date + '/' + save_name + '_' + str(i_file) + '.csv'
    summary_df.to_csv(file_path, index=False)


date = '1037'
create_folders(date)
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
# train = {'190521GongChangyang': SI_SR_TRIALS, '190523ZengJia': SI_SR_TRIALS}

"""explore number of layers and the number of neurons"""

layer_to_explore = [
    [2],
    [10],
    [50] + [10],
    [50, 50] + [10],
    [50, 50, 50] + [10],
    [50 for _ in range(7)] + [10],
    [50 for _ in range(15)] + [10],
    [50 for _ in range(31)] + [10],
    [50 for _ in range(63)] + [10],
    [50 for _ in range(127)] + [10],

    [2, 2, 2],
    [4, 4, 4],
    [8, 8, 8],
    [16, 16, 10],
    [32, 32, 10],
    [50, 50, 10],
    [64, 64, 10],
    [128, 128, 10],
    [256, 256, 10],
    [512, 512, 10],
    [1024, 1024, 10],
]

layer_summary_np = np.zeros([0, 4])
test_name_to_store = []
cross_vali_LR_processor = ProcessorLRModelParam(train, {}, ['l_shank'])
for hidden_layer_neurons in layer_to_explore:
    test_name = 'layer'

    if len(hidden_layer_neurons) < 10:
        for layer in hidden_layer_neurons:
            test_name = test_name + '_' + str(layer)
    else:
        test_name = test_name + str(hidden_layer_neurons[0]) + '_times_' + str(len(hidden_layer_neurons))
    test_name_to_store.append(test_name)
    cross_vali_LR_processor.set_hidden_layer_neurons(hidden_layer_neurons)
    layer_summary_np = cross_vali_LR_processor.cnn_cross_vali_layer(
        date, test_name, layer_summary_np)
    keras.backend.clear_session()
    plt.show()

layer_columns = ['pearson correlation', 'RMSE', 'mean error', 'absolute mean error']
save_summary_np(layer_summary_np, layer_columns, test_name_to_store, 'layer_summary')














