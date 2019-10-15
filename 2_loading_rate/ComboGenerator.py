from ProcessorLR import ProcessorLR
from const import SUB_NAMES, COLORS, TRIAL_NAMES, MOCAP_SAMPLE_RATE, SI_SR_TRIALS
import keras
import copy
import numpy as np
import pandas as pd
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from AllSubData import AllSubData
import os


class PeakTibiaAccModel(ProcessorLR):
    def __init__(self, train_sub_and_trials, strike_off_from_IMU=1):
        imu_locations = ['l_shank']
        self.train_sub_and_trials = train_sub_and_trials
        self.imu_locations = imu_locations
        self.sensor_sampling_fre = MOCAP_SAMPLE_RATE
        self.strike_off_from_IMU = strike_off_from_IMU
        self.do_input_norm = False
        self.do_output_norm = False
        self.param_name = 'LR'
        train_all_data = AllSubData(self.train_sub_and_trials, imu_locations, self.param_name, self.sensor_sampling_fre,
                                    self.strike_off_from_IMU)
        self.train_all_data_list = train_all_data.get_all_data(imu_cut_off_fre=60)

    def peak_tibia_acc_model(self, test_date, test_name):
        """
        The first channel is the axial tibia acceleration along the shank.
        :return:
        """
        predict_result_df, predicted_value_df = pd.DataFrame(), pd.DataFrame()
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        sample_num = len(input_list)
        trial_ids = train_all_data_list.get_trial_id_list()
        sub_id_list = train_all_data_list.get_sub_id_list()
        sub_id_set = tuple(set(sub_id_list))
        for sub_id in sub_id_set:
            input_list_sub, output_list_sub, test_trial_ids = [], [], []
            for i_sample in range(sample_num):
                if sub_id_list[i_sample] == sub_id:
                    input_list_sub.append(input_list[i_sample])
                    output_list_sub.append(output_list[i_sample])
                    test_trial_ids.append(trial_ids[i_sample])

            y_pred_sub = self.convert_input(input_list_sub, MOCAP_SAMPLE_RATE)
            y_true_sub = self.convert_output(output_list_sub)
            predict_result_df = self.save_detailed_results(predict_result_df, SUB_NAMES[sub_id], y_true_sub,
                                                           y_pred_sub, test_trial_ids)
            predicted_value_df = self.save_all_predicted_values(predicted_value_df, y_true_sub, y_pred_sub, sub_id,
                                                                test_trial_ids)
        Evaluation.export_prediction_result(predict_result_df, test_date, test_name)
        Evaluation.export_predicted_values(predicted_value_df, test_date, test_name)

    def convert_input(self, input_all_list, sampling_fre):
        step_num = len(input_all_list)
        input_array = np.zeros([step_num])
        # plt.figure()
        for i_step in range(step_num):
            axial_tibia_acc = -input_all_list[i_step][:, 0]
            input_array[i_step] = np.max(axial_tibia_acc)
        #     plt.plot(axial_tibia_acc)
        # plt.show()
        return input_array


class ComboGenerator:
    @staticmethod
    def all_combos(train, date, c51=False, c52=False, c53=False, c54=False, c55=False, do_pta=False):
        """

        :param train: dict. Train subjects and trials.
        :param date: str. Used as the head of result file names
        :param c51: bool. Whether to run through each sensor
        :param c52: bool. Whether to run through combinations of two sensors
        :param c53: bool. Whether to run through combinations of three sensors
        :param c54: bool. Whether to run through combinations of four sensors
        :param c55: bool. Whether to run using all five sensors
        :return:
        """

        ComboGenerator.create_folders(date)

        segments = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
        if c51:
            print('\n\nDoing C51')
            for segment in segments:
                print('\nCurrent segment: ' + segment)
                cross_vali_LR_processor = ProcessorLR(train, {}, [segment])
                cross_vali_LR_processor.cnn_cross_vali(test_date=date, test_name=date + '_' + segment, plot=False)
                keras.backend.clear_session()

        if c52:
            print('\n\nDoing C52')
            segment_combos = ComboGenerator.combinations_by_subset(segments, 2)
            for combo in segment_combos:
                print('\nCurrent segments: ' + str(combo))
                cross_vali_LR_processor = ProcessorLR(train, {}, combo)
                test_name = date
                for segment in combo:
                    test_name = test_name + '_' + segment
                cross_vali_LR_processor.cnn_cross_vali(test_date=date, test_name=test_name, plot=False)
                keras.backend.clear_session()

        if c53:
            print('\n\nDoing C53')
            segment_combos = ComboGenerator.combinations_by_subset(segments, 3)
            for combo in segment_combos:
                print('\nCurrent segments: ' + str(combo))
                cross_vali_LR_processor = ProcessorLR(train, {}, combo)
                test_name = date
                for segment in combo:
                    test_name = test_name + '_' + segment
                cross_vali_LR_processor.cnn_cross_vali(test_date=date, test_name=test_name, plot=False)
                keras.backend.clear_session()

        if c54:
            print('\n\nDoing C54')
            for segment in segments:
                segment_list = copy.deepcopy(segments)
                segment_list.remove(segment)
                print('\nCurrent segments: ' + str(segment_list))
                cross_vali_LR_processor = ProcessorLR(train, {}, segment_list)
                test_name = date
                for segment in segment_list:
                    test_name = test_name + '_' + segment
                cross_vali_LR_processor.cnn_cross_vali(test_date=date, test_name=test_name, plot=False)
                keras.backend.clear_session()

        if c55:
            print('\n\nDoing all segment')
            cross_vali_LR_processor = ProcessorLR(train, {}, segments)
            test_name = date
            for segment in segments:
                test_name = test_name + '_' + segment
            cross_vali_LR_processor.cnn_cross_vali(test_date=date, test_name=test_name, plot=False)
            keras.backend.clear_session()

        if do_pta:
            print('\n\nDoing PTA model LR prediction')
            pta_model = PeakTibiaAccModel(train)
            pta_model.peak_tibia_acc_model(test_date=date, test_name=date + '_pta')

    @staticmethod
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

    @staticmethod
    def combinations_by_subset(seq, r):
        if r:
            for i in range(r - 1, len(seq)):
                for cl in ComboGenerator.combinations_by_subset(seq[:i], r - 1):
                    yield cl + [seq[i], ]
        else:
            yield list()
