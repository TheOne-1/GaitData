import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from numpy import sqrt
from scipy.stats import pearsonr
from keras import optimizers
from keras.callbacks import EarlyStopping
from const import COLORS, SI_SR_TRIALS
import pandas as pd


class Evaluation:
    def __init__(self, x_train, x_test, y_train, y_test, x_train_aux=None, x_test_aux=None):
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        self._x_train_aux = x_train_aux
        self._x_test_aux = x_test_aux

    @staticmethod
    def get_all_scores(y_test, y_pred, precision=None):
        pearson_coeff = pearsonr(y_test, y_pred)[0]
        RMSE = sqrt(mean_squared_error(y_test, y_pred))
        errors = y_test - y_pred
        mean_error = np.mean(errors, axis=0)
        absolute_mean_error = np.mean(abs(errors))
        if precision:
            pearson_coeff = np.round(pearson_coeff, precision)
            RMSE = np.round(RMSE, precision)
            mean_error = np.round(mean_error, precision)
            absolute_mean_error = np.round(absolute_mean_error, precision)
        return pearson_coeff, RMSE, mean_error, absolute_mean_error

    def evaluate_nn(self, model, check_convergence=True):
        # train NN
        # lr = learning rate, the other params are default values
        optimizer = optimizers.Nadam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # val_loss = validation loss, patience is the tolerance
        early_stopping_patience = 5     # !!!
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
        # epochs is the maximum training round, validation split is the size of the validation set,
        # callback stops the training if the validation was not approved
        batch_size = 32  # the size of data that be trained together
        epoch_num = 200
        if self._x_train_aux is not None:
            r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                          batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
                          verbose=2)
            if np.isnan(r.history['loss'][0]):
                raise ValueError('Loss is Nan')
            n_epochs = len(r.history['loss'])
            # retrain the model if the model did not converge
            while check_convergence and n_epochs < early_stopping_patience + 3:
                print('Epcohs number was {num}, reset weights and retrain'.format(num=n_epochs))
                model.reset_states()
                r = model.fit(x={'main_input': self._x_train, 'aux_input': self._x_train_aux}, y=self._y_train,
                              batch_size=batch_size, epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping],
                              verbose=2)
                n_epochs = len(r.history['loss'])
            y_pred = model.predict(x={'main_input': self._x_test, 'aux_input': self._x_test_aux},
                                   batch_size=batch_size).ravel()
            # print('Final model, loss = {loss}, epochs = {epochs}'.format(loss=r.history['loss'][-1], epochs=len())
        else:
            model.fit(self._x_train, self._y_train, batch_size=batch_size,
                      epochs=epoch_num, validation_split=0.2, callbacks=[early_stopping])
            y_pred = model.predict(self._x_test, batch_size=batch_size).ravel()
        return y_pred

    @staticmethod
    def plot_nn_result(y_true, y_pred, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if y_true.shape != 1:
            y_true = y_true.ravel()
        if y_pred.shape != 1:
            y_pred = y_pred.ravel()

        pearson_coeff, RMSE, mean_error, _ = Evaluation.get_all_scores(y_true, y_pred, precision=3)
        plt.figure()
        plt.plot(y_true, y_pred, 'b.')
        plt.plot([0, 250], [0, 250], 'r--')
        RMSE_str = str(RMSE)
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        plt.title(title + '\np_correlation: ' + pearson_coeff + '   RMSE: '
                  + RMSE_str + '  Mean error: ' + mean_error_str)
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        return pearson_coeff, RMSE, mean_error

    @staticmethod
    def plot_nn_result_cate_color(y_true, y_pred, category_id, category_names, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        if y_true.shape != 1:
            y_true = y_true.ravel()
        if y_pred.shape != 1:
            y_pred = y_pred.ravel()
        plt.figure()
        pearson_coeff, RMSE, mean_error, _ = Evaluation.get_all_scores(y_true, y_pred, precision=3)
        RMSE_str = str(RMSE)
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        title_extended = title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str + '  Mean error: ' + mean_error_str
        plt.title(title_extended)
        print(title_extended)
        plt.plot([0, 250], [0, 250], 'r--')
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
            plt_handle, = plt.plot(y_true[category_index], y_pred[category_index], plot_pattern,
                                   color=COLORS[category_id])
            plot_handles.append(plt_handle)
        plt.legend(plot_handles, plot_names)
        plt.xlabel('true value')
        plt.ylabel('predicted value')

    @staticmethod
    def plot_continuous_result(y_true, y_pred, title=''):
        # change the shape of data so that no error will be raised during pearsonr analysis
        pearson_coeff, RMSE, mean_error, _ = Evaluation.get_all_scores(y_true, y_pred, precision=3)
        plt.figure()
        plot_true, = plt.plot(y_true[:2000])
        plot_pred, = plt.plot(y_pred[:2000])
        RMSE_str = str(RMSE)
        mean_error_str = str(mean_error)
        pearson_coeff = str(pearsonr(y_true, y_pred))[1:6]
        plt.title(title + '\ncorrelation: ' + pearson_coeff + '   RMSE: ' + RMSE_str +
                  '  Mean error: ' + mean_error_str)
        plt.legend([plot_true, plot_pred], ['true values', 'predicted values'])
        plt.xlabel('Sample number')
        plt.ylabel('GRF (body weight)')

    @staticmethod
    def reset_weights(model):
        model.reset_states()

    @staticmethod
    def insert_prediction_result(predict_result_df, sub_name, pearson_coeff, RMSE, mean_error):
        sub_df = pd.DataFrame([[sub_name, pearson_coeff, RMSE, mean_error]])
        predict_result_df = predict_result_df.append(sub_df)
        return predict_result_df

    @staticmethod
    def export_prediction_result(predict_result_df, test_date, test_name):
        column_names = ['subject name', 'parameter name']
        column_names.extend(SI_SR_TRIALS)
        column_names.extend(['All trials'])
        predict_result_df.columns = column_names

        for param_name in ['pearson correlation', 'RMSE', 'mean error', 'absolute mean error']:
            param_df = predict_result_df[predict_result_df['parameter name'] == param_name]
            result_abs_mean = np.mean(abs(param_df.iloc[:, 2:]))
            mean_value_list = ['absolute mean', param_name]
            mean_value_list.extend(result_abs_mean.tolist())
            predict_result_df.loc[-1] = mean_value_list
            predict_result_df = predict_result_df.reset_index(drop=True)

        file_path = 'result_conclusion/' + test_date + '/trial_summary/' + test_name + '.xlsx'
        predict_result_df.to_excel(file_path, index=False)

    @staticmethod
    def export_predicted_values(predicted_value_df, test_date, test_name):
        predicted_value_df.columns = ['subject id', 'trial id', 'true LR', 'predicted LR']
        file_path = 'result_conclusion/' + test_date + '/step_result/' + test_name + '.csv'
        predicted_value_df.to_csv(file_path, index=False)

