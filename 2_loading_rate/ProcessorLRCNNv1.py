"""
Each channel was processed independently
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from ProcessorLR import ProcessorLR
from keras.models import Sequential, Model


class ProcessorLRCNNv1(ProcessorLR):
    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm
        """
        step_num = len(input_all_list)
        sample_before_strike = int(12 * (sampling_fre / 100))
        sample_after_strike = int(8 * (sampling_fre / 100))
        win_len = sample_after_strike + sample_before_strike        # convolution kernel length
        step_input = np.zeros([step_num, win_len, 2])
        for i_step in range(step_num):
            acc_data = input_all_list[i_step][:, 2:3]
            gyr_data = input_all_list[i_step][:, 3:4]
            strike_data = input_all_list[i_step][:, 6]
            strike_sample_num = np.where(strike_data == 1)[0][0]
            start_sample = strike_sample_num - sample_before_strike
            end_sample = strike_sample_num + sample_after_strike
            step_data = np.column_stack([acc_data[start_sample:end_sample, :], gyr_data[start_sample:end_sample, :]])
            step_input[i_step, :, :] = step_data
        feature_names = None
        return step_input, feature_names

    def cnn_solution(self):
        input_shape = self._x_train.shape
        feature_outputs = []
        inputs = Input((input_shape[1:]))
        multiple_features = Lambda(ProcessorLR.splitter)(inputs)
        # for each feature
        for feature in multiple_features:
            # feature_output = Conv1D(filters=100, kernel_size=input_shape[1])(feature)
            feature_output = Conv1D(filters=5, kernel_size=int(input_shape[1]/2))(feature)
            feature_outputs.append(feature_output)
        joined_outputs = concatenate(feature_outputs)
        joined_outputs = Activation('relu')(joined_outputs)
        outputs = Flatten()(joined_outputs)
        outputs = Dense(40, activation='relu')(outputs)
        outputs = Dense(20, activation='relu')(outputs)
        outputs = Dense(1, activation='relu')(outputs)
        model = Model(inputs, outputs)
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        y_pred = my_evaluator.evaluate_nn(model)
        my_evaluator.plot_nn_result(self._y_test, y_pred, 'loading rate')
        plt.show()

