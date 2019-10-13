from ProcessorLR import ProcessorLR
from const import SUB_NAMES, COLORS, TRIAL_NAMES, MOCAP_SAMPLE_RATE, SI_SR_TRIALS
import keras
import copy
import numpy as np
import pandas as pd
from Evaluation import Evaluation
import matplotlib.pyplot as plt


class PeakTibiaAccModel(ProcessorLR):
    def __init__(self, train_sub_and_trials, strike_off_from_IMU=1):
        imu_locations = ['l_shank']
        super().__init__(train_sub_and_trials, {}, imu_locations, strike_off_from_IMU=strike_off_from_IMU,
                         do_input_norm=False, do_output_norm=False)

    def peak_tibia_acc_model(self, test_name):
        """
        The first channel is the axial tibia acceleration along the shank.
        :return:
        """
        predict_result_df = pd.DataFrame()
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
            Evaluation.plot_nn_result_cate_color(y_true_sub, y_pred_sub, test_trial_ids, TRIAL_NAMES, SUB_NAMES[sub_id])
            predict_result_df = self.save_detailed_results(predict_result_df, SUB_NAMES[sub_id], y_true_sub,
                                                           y_pred_sub, test_trial_ids)
        Evaluation.export_prediction_result(predict_result_df, test_name)
        plt.show()

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
        segments = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
        if c51:
            print('\n\nDoing C51')
            for segment in segments:
                print('\nCurrent segment: ' + segment)
                cross_vali_LR_processor = ProcessorLR(train, {}, [segment])
                cross_vali_LR_processor.cnn_cross_vali(test_name=date + '_' + segment, plot=False)
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
                cross_vali_LR_processor.cnn_cross_vali(test_name=test_name, plot=False)
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
                cross_vali_LR_processor.cnn_cross_vali(test_name=test_name, plot=False)
                keras.backend.clear_session()

        if c54:
            print('\n\nDoing C54')
            for segment in segments[2:]:
                segment_list = copy.deepcopy(segments)
                segment_list.remove(segment)
                print('\nCurrent segments: ' + str(segment_list))
                cross_vali_LR_processor = ProcessorLR(train, {}, segment_list)
                test_name = date
                for segment in segment_list:
                    test_name = test_name + '_' + segment
                cross_vali_LR_processor.cnn_cross_vali(test_name=test_name, plot=False)
                keras.backend.clear_session()

        if c55:
            print('\n\nDoing all segment')
            cross_vali_LR_processor = ProcessorLR(train, {}, segments)
            test_name = date
            for segment in segments:
                test_name = test_name + '_' + segment
            cross_vali_LR_processor.cnn_cross_vali(test_name=test_name, plot=False)
            keras.backend.clear_session()

        if do_pta:
            print('\n\nDoing PTA model LR prediction')
            pta_model = PeakTibiaAccModel(train)
            pta_model.peak_tibia_acc_model(test_name=date + '_pta')


    @staticmethod
    def combinations_by_subset(seq, r):
        if r:
            for i in range(r - 1, len(seq)):
                for cl in ComboGenerator.combinations_by_subset(seq[:i], r - 1):
                    yield cl + [seq[i], ]
        else:
            yield list()
