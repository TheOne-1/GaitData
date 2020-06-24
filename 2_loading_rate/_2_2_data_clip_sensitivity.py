from ProcessorLR import ProcessorLR
from const import SUB_AND_SI_SR_TRIALS, TRIAL_NAMES, SUB_NAMES, SI_SR_TRIALS, RUNNING_TRIALS
import copy
import matplotlib.pyplot as plt
import keras
from Evaluation import Evaluation
import numpy as np
import pandas as pd
from ComboGenerator import ComboGenerator
import os


class ProcessorLRDataClip(ProcessorLR):

    def cnn_cross_vali_clip(self, test_date, test_name, clip_start, clip_end, clip_summary_np, test_set_sub_num=1):
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

            self._x_train, self._x_train_aux = self.convert_input(input_list_train, self.sensor_sampling_fre,
                                                                  clip_start, clip_end)
            self._y_train = ProcessorLR.convert_output(output_list_train)
            self._x_test, self._x_test_aux = self.convert_input(input_list_test, self.sensor_sampling_fre, clip_start,
                                                                clip_end)
            self._y_test = ProcessorLR.convert_output(output_list_test)

            self.do_normalization()
            self.define_cnn_model()

            my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                      self._x_test_aux)
            y_pred = my_evaluator.evaluate_nn(self.model)
            if self.do_output_norm:
                y_pred = self.norm_output_reverse(y_pred)

            predict_result_df = self.save_detailed_results(predict_result_df, SUB_NAMES[test_id_list[0]],
                                                           self._y_test, y_pred, test_trial_ids)
            predicted_value_df = self.save_all_predicted_values(predicted_value_df, self._y_test, y_pred,
                                                                test_id_list[0], test_trial_ids)
        predict_result_df = Evaluation.export_prediction_result(predict_result_df, test_date, test_name)
        Evaluation.export_predicted_values(predicted_value_df, test_date, test_name)

        all_sub_summary_df = predict_result_df[predict_result_df['subject name'] == 'absolute mean']
        clip_values = np.concatenate([[clip_start], [clip_end], all_sub_summary_df['All trials'].values.reshape([-1])])
        clip_summary_np = np.row_stack([clip_summary_np, clip_values])
        return clip_summary_np


date = '1030'
show_result = False

ComboGenerator.create_folders(date)
segment = 'l_shank'
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
# train = {'190521GongChangyang': SI_SR_TRIALS, '190523ZengJia': SI_SR_TRIALS}

cross_vali_LR_processor = ProcessorLRDataClip(train, {}, [segment], strike_off_from_IMU=2)

clip_summary_np = np.zeros([0, 6])
clip_end = 30
for clip_start in range(-24, 13, 4):
    clip_summary_np = cross_vali_LR_processor.cnn_cross_vali_clip(
        date, 'start_' + str(clip_start) + '_end_' + str(clip_end), clip_start, clip_end, clip_summary_np)
    keras.backend.clear_session()

clip_start = -12
for clip_end in range(2, 35, 4):
    clip_summary_np = cross_vali_LR_processor.cnn_cross_vali_clip(
        date, 'start_' + str(clip_start) + '_end_' + str(clip_end), clip_start, clip_end, clip_summary_np)
    keras.backend.clear_session()

clip_summary_df = pd.DataFrame(clip_summary_np)
clip_summary_df.columns = ['clip start', 'clip end', 'pearson correlation', 'RMSE', 'mean error', 'absolute mean error']
file_path = 'result_conclusion/' + date + '/clip_summary.csv'
i_file = 0
while os.path.isfile(file_path):
    i_file += 1
    file_path = 'result_conclusion/' + date + '/clip_summary_' + str(i_file) + '.csv'
clip_summary_df.to_csv(file_path, index=False)








