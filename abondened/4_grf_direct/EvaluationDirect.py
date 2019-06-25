import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from numpy import sqrt
from scipy.stats import pearsonr
from keras import optimizers
from keras.callbacks import EarlyStopping
from const import COLORS, TRIAL_NAMES
from keras import backend as K


class EvaluationDirect:
    def __init__(self, x_train, x_test, y_train, y_test, x_train_aux=None, x_test_aux=None):
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._x_train_aux = x_train_aux
        self._x_test_aux = x_test_aux

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
        R2, RMSE, mean_error = EvaluationDirect._get_all_scores(self._y_test, y_pred, precision=3)

        plt.figure()
        plt.plot(self._y_test[:1000])
        plt.plot(y_pred[:1000])
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(self._y_test, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')

    def evaluate_nn(self, model):
        # train NN
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # val_loss = validation loss, patience is the tolerance
        early_stopping_patience = 5
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        batch_size = 20  # the size of data that be trained together
        if self._x_train_aux is not None:
            r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                          batch_size=batch_size, epochs=200, validation_split=0.2, callbacks=[early_stopping], verbose=2)
            n_epochs = len(r.history['loss'])
            # retrain the model if the model did not converge
            while n_epochs < early_stopping_patience + 5:
                print('Epcohs number was {num}, reset weights and retrain'.format(num=n_epochs))
                EvaluationDirect.reset_weights(model)
                r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                              batch_size=batch_size, epochs=50, validation_split=0.2, callbacks=[early_stopping],
                              verbose=2)
                n_epochs = len(r.history['loss'])
            y_pred = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
                                   batch_size=batch_size).ravel()
            # print('Final model, loss = {loss}, epochs = {epochs}'.format(loss=r.history['loss'][-1], epochs=len())
        else:
            model.fit(self._x_train, self._y_train, batch_size=batch_size,
                      epochs=200, validation_split=0.2, callbacks=[early_stopping])
            y_pred = model.predict(self._x_test, batch_size=batch_size).ravel()
        return y_pred

    @staticmethod
    def plot_nn_result(y_true, y_pred, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if y_true.shape != 1:
            y_true = y_true.ravel()
        if y_pred.shape != 1:
            y_pred = y_pred.ravel()

        R2, RMSE, mean_error = EvaluationDirect._get_all_scores(y_true, y_pred, precision=3)
        plt.figure()
        plt.plot(y_true, y_pred, 'b.')
        plt.plot([0, 2.5], [0, 2.5], 'r--')
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')

    @staticmethod
    def plot_nn_result_cate_color(y_true, y_pred, category_id, category_names, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if y_true.shape != 1:
            y_true = y_true.ravel()
        if y_pred.shape != 1:
            y_pred = y_pred.ravel()
        R2, RMSE, mean_error = EvaluationDirect._get_all_scores(y_true, y_pred, precision=3)
        plt.figure()
        plt.plot([0, 2.5], [0, 2.5], 'r--')
        category_list = set(category_id)
        category_id_array = np.array(category_id)
        plot_handles, plot_names = [], []
        for category_id in list(category_list):
            category_name = category_names[category_id]
            plot_names.append(category_name)
            if 'mini' in category_name:
                plot_pattern = 'x'
            else:
                plot_pattern = '.'
            category_index = np.where(category_id_array == category_id)[0]
            plt_handle, = plt.plot(y_true[category_index], y_pred[category_index], plot_pattern, color=COLORS[category_id])
            plot_handles.append(plt_handle)
        plt.legend(plot_handles, plot_names)
        RMSE_str = str(RMSE[0])
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')

    @staticmethod
    def reset_weights(model):
        session = K.get_session()
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)











