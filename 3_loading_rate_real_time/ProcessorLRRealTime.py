from Evaluation import Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from AllSubDataRealTime import AllSubDataRealTime
from keras.layers import *
import scipy.interpolate as interpo
from sklearn.ensemble import GradientBoostingRegressor
from const import SUB_NAMES, TRIAL_NAMES, COLORS, DATA_COLUMNS_XSENS
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ProcessorLRRealTime:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre,
                 split_train=False, do_input_norm=True):
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.sensor_sampling_fre = sensor_sampling_fre
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.param_name = 'LR'
        train_all_data = AllSubDataRealTime(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre)
        self.train_all_data_list = train_all_data.get_all_data(clean=True)
        if not self.split_train:
            test_all_data = AllSubDataRealTime(self.test_sub_and_trials, self.param_name, self.sensor_sampling_fre)
            self.test_all_data_list = test_all_data.get_all_data(clean=True)

    def prepare_data(self):
        input_list, aux_list, output_list = self.train_all_data_list.get_input_output_list()
        self._x_train = self.list_to_array(input_list)
        self._x_train_aux = self.list_to_array(aux_list)
        self._y_train = self.list_to_array(output_list)
        if not self.split_train:
            input_list, aux_list, output_list = self.test_all_data_list.get_input_output_list()
            self.test_sub_id_list = self.test_all_data_list.get_sub_id_list()
            self.test_trial_id_list = self.test_all_data_list.get_trial_id_list()
            self._x_test = self.list_to_array(input_list)
            self._x_test_aux = self.list_to_array(aux_list)
            self._y_test = self.list_to_array(output_list)
        else:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._x_train_aux, self._x_test_aux, self._y_train, self._y_test =\
                train_test_split(self._x_train, self._x_train_aux, self._y_train, test_size=0.3)
        # do input normalization
        if self.do_input_norm:
            self.norm_input()

    @staticmethod
    def list_to_array(the_list):
        data_len = len(the_list)
        sample_shape = the_list[0].shape
        the_array_shape = [data_len]
        the_array_shape.extend(sample_shape)
        the_array = np.zeros(the_array_shape)
        for i_sample in range(data_len):
            if len(sample_shape) == 1:
                the_array[i_sample, :] = the_list[i_sample]
            else:
                the_array[i_sample, :, :] = the_list[i_sample]
        return the_array

    def linear_regression_solution(self):
        model = LinearRegression()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    def GBDT_solution(self):
        model = GradientBoostingRegressor()
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        print(model.feature_importances_)
        plt.show()
        return model.feature_importances_

    def nn_solution(self):
        model = MLPRegressor(hidden_layer_sizes=40, activation='logistic')
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    #function to split the input in multiple outputs
    @staticmethod
    def splitter(x):
        feature_num = x.shape[2]
        return [x[:, :, i:i+1] for i in range(feature_num)]

    def norm_input(self):
        main_input_scalar = StandardScaler()
        channel_num = self._x_train.shape[2]
        for i_channel in range(channel_num):
            self._x_train[:, :, i_channel] = main_input_scalar.fit_transform(self._x_train[:, :, i_channel])
            self._x_test[:, :, i_channel] = main_input_scalar.transform(self._x_test[:, :, i_channel])

        if hasattr(self, '_x_train_aux'):
            # MinMaxScaler is more suitable because StandardScalar will make the input greatly differ from each other
            aux_input_scalar = MinMaxScaler()
            self._x_train_aux = aux_input_scalar.fit_transform(self._x_train_aux)
            self._x_test_aux = aux_input_scalar.transform(self._x_test_aux)






















