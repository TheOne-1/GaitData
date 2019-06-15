from EvaluationDirect import EvaluationDirect
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from AllSubDataDirect import AllSubData
from keras.layers import *
import scipy.interpolate as interpo
from sklearn.ensemble import GradientBoostingRegressor
from const import SUB_NAMES, TRIAL_NAMES, COLORS, DATA_COLUMNS_XSENS
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ProcessorDirect:
    def __init__(self, train_sub_and_trials, test_sub_and_trials, sensor_sampling_fre, strike_off_from_IMU=False,
                 split_train=False, do_input_norm=True):
        self.train_sub_and_trials = train_sub_and_trials
        self.test_sub_and_trials = test_sub_and_trials
        self.sensor_sampling_fre = sensor_sampling_fre
        self.strike_off_from_IMU = strike_off_from_IMU
        self.split_train = split_train
        self.do_input_norm = do_input_norm
        self.param_name = 'LR'
        train_all_data = AllSubData(self.train_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
        self._x_train, self._y_train = train_all_data.get_all_data()
        if test_sub_and_trials is not None:
            test_all_data = AllSubData(self.test_sub_and_trials, self.param_name, self.sensor_sampling_fre, self.strike_off_from_IMU)
            self._x_test, self._y_test = test_all_data.get_all_data()

    def prepare_data(self):
        if self.split_train:
            # split the train, test set from the train data
            self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
                self._x_train, self._y_train, test_size=0.33)
        # do input normalization
        if self.do_input_norm:
            self.norm_input()

    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        # this method has to be overwritten
        raise NotImplementedError('this convert_step_input method has to be overwritten')

    def linear_regression_solution(self):
        model = LinearRegression()
        my_evaluator = EvaluationDirect(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    def GBDT_solution(self):
        model = GradientBoostingRegressor()
        my_evaluator = EvaluationDirect(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        print(model.feature_importances_)
        plt.show()
        return model.feature_importances_

    def nn_solution(self):
        model = MLPRegressor(hidden_layer_sizes=(20, 20, 20, 20), activation='relu', max_iter=100)
        my_evaluator = EvaluationDirect(self._x_train, self._x_test, self._y_train, self._y_test)
        my_evaluator.evaluate_sklearn(model, 'loading rate')
        plt.show()

    #function to split the input in multiple outputs
    @staticmethod
    def splitter(x):
        feature_num = x.shape[2]
        return [x[:, :, i:i+1] for i in range(feature_num)]

    @staticmethod
    def resample_channel(data_array, resampled_len):
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        data_len = data_array.shape[1]
        data_step = np.arange(0, data_len)
        resampled_step = np.linspace(0, data_len, resampled_len)
        tck, data_step = interpo.splprep(data_array, u=data_step, s=0)
        data_resampled = interpo.splev(resampled_step, tck, der=0)[0]
        return data_resampled

    def norm_input(self):
        main_input_scalar = StandardScaler()
        self._x_train = main_input_scalar.fit_transform(self._x_train)
        self._x_test = main_input_scalar.transform(self._x_test)

        if hasattr(self, '_x_train_aux'):
            # MinMaxScaler is more suitable because StandardScalar will make the input greatly differ from each other
            aux_input_scalar = MinMaxScaler()
            self._x_train_aux = aux_input_scalar.fit_transform(self._x_train_aux)
            self._x_test_aux = aux_input_scalar.transform(self._x_test_aux)





