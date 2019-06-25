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
from keras.applications.inception_v3 import InceptionV3
from keras import backend
from AllSubData import AllSubData
from keras.models import load_model


class ProcessorLRLoadModel(ProcessorLR):
    def __init__(self, test_sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=False,
                 split_train=False, do_input_norm=True, do_output_norm=True):
        self.test_sub_and_trials = test_sub_and_trials
        self.sensor_sampling_fre = sensor_sampling_fre
        self.strike_off_from_IMU = strike_off_from_IMU
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.do_output_norm = do_output_norm
        self.param_name = 'LR'
        test_all_data = AllSubData(self.test_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
        self.test_all_data_list = test_all_data.get_all_data()

    def prepare_data(self):
        test_all_data_list = ProcessorLR.clean_all_data(self.test_all_data_list)
        input_list, output_list = test_all_data_list.get_input_output_list()
        self._x_test, self._x_test_aux = self.convert_input(input_list, self.sensor_sampling_fre)
        self._y_test = ProcessorLR.convert_output(output_list)

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
        aux_input = ProcessorLRLoadModel.clean_aux_input(aux_input)
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
        model = load_model('lr_model.h5')
        y_pred = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux}).ravel()
        for result in y_pred:
            print(result)
        Evaluation.plot_nn_result(self._y_test, y_pred, 'loading rate')
        plt.show()

    def save_model(self):
        # # for test
        # test_data = self._x_test[0:1, :, :]
        # test_result = self.model.predict(test_data)
        # print(test_result)
        self.model.save('lr_model.h5', include_optimizer=False)
        convert('lr_model.h5', 'fdeep_model.json')


