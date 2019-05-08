import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from numpy import sqrt
from scipy.stats import pearsonr


class Evaluation:
    def __init__(self, x_train, x_test, y_train, y_test):
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test

    @staticmethod
    def _get_all_scores(y_test, y_pred, precision=None):
        R2 = r2_score(y_test, y_pred, multioutput='raw_values')
        RMSE = sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        errors = y_test - y_pred
        mean_error = np.mean(errors, axis=0)
        if precision:
            R2 = np.round(R2, precision)
            RMSE = np.round(RMSE, precision)
            mean_error = np.round(mean_error, precision)
        return R2, RMSE, mean_error

    def evaluate_sklearn(self, model, title=''):
        model.fit(self._x_train, self._y_train)
        y_pred = model.predict(self._x_test)
        y_pred = y_pred.reshape(-1, 1)
        R2, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred, precision=3)
        # plot
        for i_plot in range(y_pred.shape[1]):
            plt.figure()
            plt.plot(self._y_test[:, i_plot], y_pred[:, i_plot], 'b.')
            RMSE_str = str(RMSE[i_plot])
            mean_error_str = str(mean_error[i_plot])
            pearson_coeff = str(pearsonr(self._y_test[:, i_plot], y_pred[:, i_plot]))[1:6]
            plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                      '  Mean error: ' + mean_error_str)
            plt.xlabel('true value')
            plt.ylabel('predicted value')


class EvaluationWhiteBox:
    @staticmethod
    def white_box_FPA(true_value, acc_angle_mean, yaw_rotation, roll_rotation=0):
        pred_value = acc_angle_mean - 0.5*yaw_rotation



















