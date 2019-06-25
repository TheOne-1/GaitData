from Evaluation import Evaluation
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from ProcessorLR import ProcessorLR
from convert_model import convert


class ProcessorLRCNNv0(ProcessorLR):
    # convert the input from list to ndarray
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm
        """
        step_num = len(input_all_list)
        sample_before_strike = int(1 * (sampling_fre / 100))
        sample_after_strike = int(1 * (sampling_fre / 100))
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
        model = Sequential()
        # debug kernel_size = 1, 看下下一层有几个输入
        # deliberately set kernel_size equal to input_shape[0] so that
        input_shape = self._x_train.shape
        model.add(Conv1D(filters=10, kernel_size=input_shape[1], input_shape=input_shape[1:]))
        model.add(Flatten())
        model.add(Dense(80, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(1, activation='linear'))
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test)
        y_pred = my_evaluator.evaluate_nn(model)

        test_data = np.array([[[0.13686751, 0.22877938], [0.63733118, 0.0446376]], [[0.13686751, 0.22877938], [0.63733118, 0.0446376]]])
        test_result = model.predict(test_data)
        print(test_result)
        model.save('lr_model.h5', include_optimizer=False)
        convert('lr_model.h5', 'fdeep_model.json')

        my_evaluator.plot_nn_result(self._y_test, y_pred, 'loading rate')
        plt.show()

