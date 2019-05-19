import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from numpy import sqrt
from scipy.stats import pearsonr
from keras import optimizers
from keras.callbacks import EarlyStopping


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
        R2, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred, precision=3)

        plt.figure()
        plt.plot(self._y_test, y_pred, 'b.')
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(self._y_test, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')

    def evaluate_nn(self, model, title=''):
        # train NN
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # val_loss = validation loss, patience is the tolerance
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        batch_size = 200  # the size of data that be trained together
        model.fit(self._x_train, self._y_train, batch_size=batch_size,
                  epochs=100, validation_split=0.2, callbacks=[early_stopping])

        # test NN
        y_pred = model.predict(self._x_test, batch_size=batch_size).ravel()
        R2, RMSE, mean_error = Evaluation._get_all_scores(self._y_test, y_pred, precision=3)
        plt.figure()
        plt.plot(self._y_test, y_pred, 'b.')
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(self._y_test, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')











