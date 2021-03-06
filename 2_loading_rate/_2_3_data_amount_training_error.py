"""To answer the question: "It would be helpful to include the amount of data available for training models
 from each subject and information about the relative success of model training for each IMU configuration."""

from ProcessorLR import ProcessorLR
from const import SUB_AND_SI_SR_TRIALS, TRIAL_NAMES, SUB_NAMES, SI_SR_TRIALS, RUNNING_TRIALS, SUB_ID_PAPER
import copy
import matplotlib.pyplot as plt
import keras
from Evaluation import Evaluation
import numpy as np
import pandas as pd
from ComboGenerator import ComboGenerator
import os
import random

MAX_EPOCH = 200


class ProcessorLRDataAmount(ProcessorLR):
    def cnn_cross_vali_data_amount(self, test_date, test_name, percent, amount_summary_np, loss_summary_np):
        test_set_sub_num = 1
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        trial_ids = train_all_data_list.get_trial_id_list()
        self.channel_num = input_list[0].shape[1] - 1
        sub_id_list = train_all_data_list.get_sub_id_list()
        sub_id_set_tuple = tuple(set(sub_id_list))
        self.get_valid_step_num(sub_id_list)

        all_sub_loc_selected = self.bagging_from_data(sub_id_list, sub_id_set_tuple, percent)

        sample_num = len(input_list)
        sub_num = len(self.train_sub_and_trials.keys())
        folder_num = int(np.ceil(sub_num / test_set_sub_num))  # the number of cross validation times
        predict_result_df = pd.DataFrame()
        predicted_value_df = pd.DataFrame()  # save all the predicted values in case reviewer ask for more analysis
        loss_train_array = np.full([MAX_EPOCH, len(sub_id_set_tuple)], -1.0)
        loss_val_array = np.full([MAX_EPOCH, len(sub_id_set_tuple)], -1.0)
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
                elif i_sample in all_sub_loc_selected:
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
            y_pred, loss_train, loss_val = my_evaluator.evaluate_nn_report_loss(self.model)
            loss_train.reverse()
            loss_val.reverse()
            if self.do_output_norm:
                y_pred = self.norm_output_reverse(y_pred)
                loss_train = self.norm_reverse_mean_squared_error(loss_train)
                loss_val = self.norm_reverse_mean_squared_error(loss_val)
            loss_train_array[:len(loss_train), i_folder] = loss_train
            loss_val_array[:len(loss_val), i_folder] = loss_val

            predict_result_df = self.save_detailed_results(predict_result_df, SUB_NAMES[test_id_list[0]],
                                                           self._y_test, y_pred, test_trial_ids)
            predicted_value_df = self.save_all_predicted_values(predicted_value_df, self._y_test, y_pred,
                                                                test_id_list[0], test_trial_ids)
        predict_result_df = Evaluation.export_prediction_result(predict_result_df, test_date, test_name)
        Evaluation.export_predicted_values(predicted_value_df, test_date, test_name)
        self.save_loss(loss_train_array, loss_val_array, test_date, test_name)

        all_sub_summary_df = predict_result_df[predict_result_df['subject name'] == 'absolute mean']
        amount_values = np.concatenate([[percent], all_sub_summary_df['All trials'].values.reshape([-1])])
        amount_summary_np = np.row_stack([amount_summary_np, amount_values])
        loss_values = np.array([percent, np.mean(loss_train_array[0, :]), np.mean(loss_val_array[0, :])])
        loss_summary_np = np.row_stack([loss_summary_np, loss_values])

        return amount_summary_np, loss_summary_np

    def get_valid_step_num(self, sub_id_list):
        sub_id_np = np.array(sub_id_list)
        sub_data_lens = []
        for sub_id in SUB_ID_PAPER:
            sub_id_loc = np.where(sub_id_np == sub_id)[0]
            sub_data_lens.append(len(sub_id_loc))
            print(sub_data_lens[-1])
            print('{:.0f} ± {:.0f}'.format(int(np.mean(sub_data_lens)), int(np.std(sub_data_lens))))
            print('{:.0f}'.format(np.sum(sub_data_lens)))

    def save_loss(self, loss_train_array, loss_val_array, test_date, test_name):
        for the_array, prefix in zip([loss_train_array, loss_val_array], ['train_loss', 'val_loss']):
            the_df = pd.DataFrame(the_array)
            loss_file_path = 'result_conclusion/' + test_date + '/loss/' + prefix + '_' + test_name + '.csv'
            the_df.to_csv(loss_file_path, index=False)

    def norm_reverse_mean_squared_error(self, mean_squared_errors):
        scale = self.output_minmax_scalar.scale_[0]
        return [error / scale for error in mean_squared_errors]

    def bagging_from_data(self, sub_id_list, sub_id_set_tuple, percent):
        """
        Generate index for randomly selected data
        :param percent: The percentage of data to be selected
        :return: Index of selected entry
        """
        sub_id_array = np.array(sub_id_list)
        all_sub_loc_selected = []
        for sub_id in sub_id_set_tuple:
            current_sub_loc = np.where(sub_id_array == sub_id)[0]
            sub_data_len = current_sub_loc.shape[0]
            sub_sample_num = int(percent * sub_data_len)
            current_sub_loc_selected = random.sample(list(current_sub_loc), sub_sample_num)
            all_sub_loc_selected.extend(current_sub_loc_selected)
        return all_sub_loc_selected


def create_folders(date):
    # create result folder
    result_main_folder = 'result_conclusion/' + date
    result_sub_folder_0 = 'result_conclusion/' + date + '/trial_summary/'
    result_sub_folder_1 = 'result_conclusion/' + date + '/step_result/'
    result_sub_folder_2 = 'result_conclusion/' + date + '/loss/'
    if not os.path.exists(result_main_folder):
        os.makedirs(result_main_folder)
    if not os.path.exists(result_sub_folder_0):
        os.makedirs(result_sub_folder_0)
    if not os.path.exists(result_sub_folder_1):
        os.makedirs(result_sub_folder_1)
    if not os.path.exists(result_sub_folder_2):
        os.makedirs(result_sub_folder_2)


def save_summary_np(summary_np, columns, segment_name_to_store, save_name):
    summary_df = pd.DataFrame(summary_np)
    summary_df.columns = columns
    summary_df.insert(0, 'segment', segment_name_to_store)

    file_path = 'result_conclusion/' + date + '/' + save_name + '.csv'
    i_file = 0
    while os.path.isfile(file_path):
        i_file += 1
        file_path = 'result_conclusion/' + date + '/' + save_name + '_' + str(i_file) + '.csv'
    summary_df.to_csv(file_path, index=False)


date = '1032'
show_result = False
create_folders(date)

optimal_segment_list = [
    ['l_shank'],
    ['l_shank', 'l_foot'],
    ['l_thigh', 'l_shank', 'l_foot'],
    ['trunk', 'pelvis', 'l_shank', 'l_foot'],
    ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
]

segment = 'l_shank'
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
# train = {'190521GongChangyang': SI_SR_TRIALS, '190523ZengJia': SI_SR_TRIALS}

amount_summary_np = np.zeros([0, 5])
loss_summary_np = np.zeros([0, 3])
segment_name_to_store = []
for segment_list in optimal_segment_list:
    cross_vali_LR_processor = ProcessorLRDataAmount(train, {}, segment_list, strike_off_from_IMU=2)
    # for percent in [3**-4, 3**-3, 3**-2, 3**-1, 0.5, 0.7, 3**0]:
    for percent in [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]:
        test_name = segment_list[0]
        for segment in segment_list[1:]:
            test_name = test_name + '_' + segment
        segment_name_to_store.append(test_name)
        test_name = test_name + '_' + str(percent)[:6]
        amount_summary_np, loss_summary_np = cross_vali_LR_processor.cnn_cross_vali_data_amount(
            date, test_name, percent, amount_summary_np, loss_summary_np)
        keras.backend.clear_session()


amount_columns = ['percent', 'pearson correlation', 'RMSE', 'mean error', 'absolute mean error']
save_summary_np(amount_summary_np, amount_columns, segment_name_to_store, 'amount_summary')
loss_columns = ['percent', 'train_loss', 'validation_loss']
save_summary_np(loss_summary_np, loss_columns, segment_name_to_store, 'loss_summary')
