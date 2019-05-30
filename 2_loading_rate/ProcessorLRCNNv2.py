"""
Cov template
"""
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from ProcessorLR import ProcessorLR
from keras.models import Sequential, Model


class ProcessorLRCNNv2(ProcessorLR):
    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        resample_len = 100
        data_clip_start, data_clip_end = 50, 80
        self.data_clip_start, self.data_clip_end = data_clip_start, data_clip_end
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, 6])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:6]
            for i_channel in range(6):
                channel_resampled = ProcessorLR.resample_channel(acc_gyr_data[:, i_channel], resample_len)
                step_input[i_step, :, i_channel] = channel_resampled[data_clip_start:data_clip_end]

        feature_names = None
        return step_input, feature_names

    def cnn_solution(self):
        input_shape = self._x_train.shape
        inputs = Input((input_shape[1:]))
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv1D(filters=5, kernel_size=20)(inputs)
        tower_1 = MaxPool1D(pool_size=11)(tower_1)

        # for each feature, add 10 * 1 cov kernel
        tower_2 = Conv1D(filters=5, kernel_size=10)(inputs)
        tower_2 = MaxPool1D(pool_size=21)(tower_2)

        # for each feature, add 5 * 1 cov kernel
        tower_3 = Conv1D(filters=5, kernel_size=5)(inputs)
        tower_3 = MaxPool1D(pool_size=26)(tower_3)

        joined_outputs = concatenate([tower_1, tower_2, tower_3], axis=1)
        joined_outputs = Activation('relu')(joined_outputs)
        outputs = Flatten()(joined_outputs)
        outputs = Dense(40, activation='relu')(outputs)
        outputs = Dense(1, activation='relu')(outputs)
        model = Model(inputs, outputs)
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        y_pred = my_evaluator.evaluate_nn(model)
        my_evaluator.plot_nn_result(self._y_test, y_pred, 'loading rate')
        plt.show()


