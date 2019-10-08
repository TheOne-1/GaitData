import matplotlib.pyplot as plt
from AllSubData import AllSubData
import scipy.interpolate as interpo
from const import SUB_NAMES, COLORS, DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from Evaluation import Evaluation
from keras import regularizers
from keras.layers import *
from keras.models import Model
from ProcessorLR import ProcessorLR
from const import TRIAL_NAMES
import numpy as np
import keras
import pandas as pd


class ProcessorLR2DConv(ProcessorLR):
    def define_cnn_model(self):
        """
        Convolution kernel shape changed from 1D to 2D.
        :return:
        """
        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        # base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        kernel_regu = regularizers.l2(0.01)
        kernel_regu = None
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv2D(filters=12, kernel_size=(35, 1), kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPooling2D(pool_size=(16, main_input_shape[2]+1-1))(tower_1)

        # for each feature, add 20 * 1 cov kernel
        tower_2 = Conv2D(filters=12, kernel_size=(20, 1), kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPooling2D(pool_size=(31, main_input_shape[2]+1-1))(tower_2)

        # for each feature, add 20 * 1 cov kernel
        tower_3 = Conv2D(filters=12, kernel_size=(10, 1), kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPooling2D(pool_size=(41, main_input_shape[2]+1-1))(tower_3)

        # for each feature, add 20 * 1 cov kernel
        tower_4 = Conv2D(filters=12, kernel_size=(3, 1), kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPooling2D(pool_size=(48, main_input_shape[2]+1-1))(tower_4)

        # for each feature, add 20 * 1 cov kernel
        tower_5 = Conv2D(filters=20, kernel_size=(1, main_input_shape[2]), kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPooling2D(pool_size=(50, 1))(tower_5)

        joined_outputs = concatenate([tower_1, tower_2, tower_3, tower_4, tower_5], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(10, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model


class ProcessorLR2DInception(ProcessorLR):

    @staticmethod
    def reshape_imu_data(data_ori, imu_axis_num=6):
        """
        Each IMU's data are transformed into an independent channel.
        :return:
        """
        data_shape = data_ori.shape
        imu_num = int(data_shape[2] / imu_axis_num)
        data_trans = np.zeros([data_shape[0], data_shape[1], imu_axis_num, imu_num])
        for i_imu in range(imu_num):
            data_trans[:, :, :, i_imu] = data_ori[:, :, i_imu*imu_axis_num:(1+i_imu)*imu_axis_num, 0]
        return data_trans

    def define_cnn_model(self):
        """
        Modified Inception V3 model. Referred to https://blog.csdn.net/zzc15806/article/details/83447006
        :return:
        """
        self._x_train = self.reshape_imu_data(self._x_train)
        self._x_test = self.reshape_imu_data(self._x_test)
        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        kernel_regu = regularizers.l2(0.01)

        # tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(main_input)
        tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(main_input)

        # tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(main_input)
        tower_2 = Conv2D(64, (21, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(main_input)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(main_input)
        tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

        joint_output_1 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_1)
        tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_1)
        tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer=kernel_regu)(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(joint_output_1)
        tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

        joint_output_2 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        tower_1 = Conv2D(1, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_2)
        tower_1 = Conv2D(1, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(1, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_2)
        tower_2 = Conv2D(1, (5, 5), padding='same', activation='relu', kernel_regularizer=kernel_regu)(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(joint_output_2)
        tower_3 = Conv2D(1, (1, 1), padding='same', activation='relu')(tower_3)

        joint_output_3 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
        # joint_output_3 = MaxPooling2D((50, 1), strides=(1, 1), padding='valid')(joint_output_3)
        # joint_output_3 = MaxPooling2D((5, 5), strides=(5, 5), padding='valid')(joint_output_3)
        joint_output_3 = MaxPooling2D((3, 3), strides=(3, 3), padding='valid')(joint_output_3)

        main_outputs = Activation('relu')(joint_output_3)
        main_outputs = Flatten()(main_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(10, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model


class ProcessorLR2DMultiLayer(ProcessorLR):
    def define_cnn_model(self):
        """
        Convolution kernel shape changed from 1D to 2D.
        Try to use multi-layer small conv kernel to replace large conv kernerl.
        Achieved 0.958 correlation. The accuracy is still largely based on initialization.
        :return:
        """
        # inception_model = InceptionV3(include_top=False, weights=None, input_shape=(299, 299, 3))

        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        kernel_regu = regularizers.l2(0.01)

        tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(main_input)
        tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(main_input)
        tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=kernel_regu)(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(main_input)
        tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

        joint_output_1 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_1)
        tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_1)
        tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer=kernel_regu)(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(joint_output_1)
        tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

        joint_output_2 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        tower_1 = Conv2D(1, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_2)
        tower_1 = Conv2D(1, (3, 3), padding='same', activation='relu')(tower_1)

        tower_2 = Conv2D(1, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(joint_output_2)
        tower_2 = Conv2D(1, (5, 5), padding='same', activation='relu', kernel_regularizer=kernel_regu)(tower_2)

        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(joint_output_2)
        tower_3 = Conv2D(1, (1, 1), padding='same', activation='relu')(tower_3)

        joint_output_3 = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)

        joint_output_3 = MaxPooling2D((3, 3), strides=(3, 3), padding='valid')(joint_output_3)

        joint_output_3 = Activation('relu')(joint_output_3)
        main_outputs = Flatten()(joint_output_3)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(150, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(10, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model

    @staticmethod
    def inception_fig(kernel_regu):
        def f(input):
            tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(input)
            tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

            tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu', kernel_regularizer=kernel_regu)(input)
            tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=kernel_regu)(tower_2)

            tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', kernel_regularizer=kernel_regu)(input)
            tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

            return keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
        return f

    @staticmethod
    def dim_reduction(kernel_regu):
        def f(input):
            conv_a1 = Conv2D(64, (1, 1), kernel_regularizer=kernel_regu)(input)
            conv_a2 = Conv2D(96, (3, 3), kernel_regularizer=kernel_regu)(conv_a1)
            conv_a3 = Conv2D(96, (3, 3), kernel_regularizer=kernel_regu, subsample=(2, 2), border_mode="valid")(conv_a2)

            # another inconsistency between model.txt and the paper
            # the Fig 10 in the paper shows a 1x1 convolution before
            # the 3x3. Going with model.txt
            conv_b = Conv2D(384, (3, 3), kernel_regularizer=kernel_regu, subsample=(2, 2), border_mode="valid")(input)
            pool_c = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(input)
            return keras.layers.concatenate([conv_a3, conv_b, pool_c], axis=1)
        return f


class ProcessorLRNoResample(ProcessorLR):
    def convert_input(self, input_all_list, sampling_fre, data_clip_start=-20, data_clip_end=10):
        """
        input start from strike-20 to strike+20
        """
        step_num = len(input_all_list)
        # data_clip_start, data_clip_end = -40, 25
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, self.channel_num])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:self.channel_num]
            step_len = acc_gyr_data.shape[0]
            aux_input[i_step, 0] = step_len
            strike_sample_num = int(np.where(input_all_list[i_step][:, -1] == 1)[0])
            aux_input[i_step, 1] = strike_sample_num
            step_input[i_step, :, :] = acc_gyr_data[strike_sample_num+data_clip_start:strike_sample_num+data_clip_end, :]

        aux_input = ProcessorLR.clean_aux_input(aux_input)
        return step_input, aux_input

    def define_cnn_model(self):
        """
        Designed for grid search.
        :return:
        """
        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        # base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        # kernel_regu = regularizers.l2(0.01)
        kernel_regu = None

        # for each feature, add 35 * 1 cov kernel
        kernel_size = np.array([3, main_input_shape[2]])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_1 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_1 = MaxPooling2D(pool_size=pool_size)(tower_1)

        kernel_size = np.array([20, 1])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_2 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPooling2D(pool_size=pool_size)(tower_2)

        kernel_size = np.array([10, 1])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_3 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPooling2D(pool_size=pool_size)(tower_3)

        kernel_size = np.array([3, 1])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_4 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPooling2D(pool_size=pool_size)(tower_4)

        # for each feature, add 20 * 1 cov kernel
        kernel_size = np.array([1, main_input_shape[2]])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_5 = Conv2D(filters=20, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPooling2D(pool_size=pool_size)(tower_5)

        joined_outputs = concatenate([tower_1, tower_2, tower_3, tower_4, tower_5], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        main_outputs = Dense(50, activation='relu')(main_outputs)
        main_outputs = Dense(50, activation='relu')(main_outputs)
        main_outputs = Dense(10, activation='relu')(main_outputs)
        main_outputs = Dense(1, activation='linear')(main_outputs)
        model = Model(inputs=main_input, outputs=main_outputs)
        self.model = model


class ProcessorLRNoResampleGridSearch(ProcessorLR):

    def convert_input(self, input_all_list, sampling_fre, data_clip_start=-40, data_clip_end=25):
        """
        input start from strike-20 to strike+20
        """
        step_num = len(input_all_list)
        # data_clip_start, data_clip_end = -40, 25
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, self.channel_num])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:self.channel_num]
            step_len = acc_gyr_data.shape[0]
            aux_input[i_step, 0] = step_len
            strike_sample_num = int(np.where(input_all_list[i_step][:, -1] == 1)[0])
            aux_input[i_step, 1] = strike_sample_num
            step_input[i_step, :, :] = acc_gyr_data[strike_sample_num+data_clip_start:strike_sample_num+data_clip_end, :]

        aux_input = ProcessorLR.clean_aux_input(aux_input)
        return step_input, aux_input

    def define_cnn_model(self):
        """
        Designed for grid search.
        :return:
        """
        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        # base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        kernel_regu = None

        # # for each feature, add 35 * 1 cov kernel
        # kernel_size = np.array([35, 1])
        # pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        # tower_1 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        # tower_1 = MaxPooling2D(pool_size=pool_size)(tower_1)

        kernel_size = np.array([3, main_input_shape[2]])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_2 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPooling2D(pool_size=pool_size)(tower_2)

        kernel_size = np.array([10, 1])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_3 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPooling2D(pool_size=pool_size)(tower_3)

        kernel_size = np.array([3, 1])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_4 = Conv2D(filters=12, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_4 = MaxPooling2D(pool_size=pool_size)(tower_4)

        # for each feature, add 20 * 1 cov kernel
        kernel_size = np.array([1, main_input_shape[2]])
        pool_size = main_input_shape[1:3] + np.array([1, 1]) - kernel_size
        tower_5 = Conv2D(filters=20, kernel_size=kernel_size, kernel_regularizer=kernel_regu)(main_input)
        tower_5 = MaxPooling2D(pool_size=pool_size)(tower_5)

        joined_outputs = concatenate([tower_2, tower_3, tower_4, tower_5], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(10, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model

    def cnn_train_test(self):
        """
        The very basic condition, use the train set to train and use the test set to test.
        :return:
        """
        predict_result_df = pd.DataFrame()
        result = np.zeros([20, 1])
        i_row, i_col = -1, -1
        for data_clip_start in range(-40, 0, 2):
            i_row += 1
            i_col = -1
            for data_clip_end in range(10, 11, 1):
                print('start: ' + str(data_clip_start) + '  end: ' + str(data_clip_end))
                i_col += 1
                self.prepare_data(data_clip_start, data_clip_end)
                self.do_normalization()
                self.define_cnn_model()
                y_pred = self.evaluate_cnn_model()
                pearson_coeff, RMSE, mean_error = Evaluation.get_all_scores(self._y_test, y_pred, precision=3)
                predict_result_df = Evaluation.insert_prediction_result(
                    predict_result_df, SUB_NAMES[0], pearson_coeff, data_clip_start, data_clip_end)

                result[i_row, i_col] = pearson_coeff

        Evaluation.export_prediction_result(predict_result_df)
        plt.imshow(result, cmap='RdBu')
        # ax = plt.gca()
        # ax.set_yticklabels([x for x in range(-50, -30, 10)])
        # ax.set_xticklabels([x for x in range(10, 30, 10)])
        plt.show()

    def evaluate_cnn_model(self):
        my_evaluator = Evaluation(self._x_train, self._x_test, self._y_train, self._y_test, self._x_train_aux,
                                  self._x_test_aux)
        y_pred = my_evaluator.evaluate_nn(self.model)
        if self.do_output_norm:
            y_pred = self.norm_output_reverse(y_pred)
        return y_pred

    def prepare_data(self, data_clip_start, data_clip_end):
        """Overwriting this function for grid search"""
        train_all_data_list = ProcessorLR.clean_all_data(self.train_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = train_all_data_list.get_input_output_list()
        self.channel_num = input_list[0].shape[1] - 1
        self._x_train, self._x_train_aux = self.convert_input(input_list, self.sensor_sampling_fre, data_clip_start, data_clip_end)
        self._y_train = ProcessorLR.convert_output(output_list)

        test_all_data_list = ProcessorLR.clean_all_data(self.test_all_data_list, self.sensor_sampling_fre)
        input_list, output_list = test_all_data_list.get_input_output_list()
        self.test_sub_id_list = test_all_data_list.get_sub_id_list()
        self.test_trial_id_list = test_all_data_list.get_trial_id_list()
        self._x_test, self._x_test_aux = self.convert_input(input_list, self.sensor_sampling_fre, data_clip_start, data_clip_end)
        self._y_test = ProcessorLR.convert_output(output_list)


class ProcessorLRCrazyKernel(ProcessorLR):
    def define_cnn_model(self):
        """
        This idea failed.
        Only one kind of big kernels and two kinds of small kernels
        :return:
        """
        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        # base_size = int(self.sensor_sampling_fre*0.01)

        # kernel_init = 'lecun_uniform'
        kernel_regu = None
        # kernel_regu = None
        # for each feature, add 20 * 1 cov kernel
        tower_1 = Conv2D(filters=30, kernel_size=(50, main_input_shape[2]), kernel_regularizer=kernel_regu)(main_input)

        # for each feature, add 20 * 1 cov kernel
        tower_2 = Conv2D(filters=30, kernel_size=(3, main_input_shape[2]), kernel_regularizer=kernel_regu)(main_input)
        tower_2 = MaxPooling2D(pool_size=(48, 1))(tower_2)

        # for each feature, add 20 * 1 cov kernel
        tower_3 = Conv2D(filters=30, kernel_size=(1, main_input_shape[2]), kernel_regularizer=kernel_regu)(main_input)
        tower_3 = MaxPooling2D(pool_size=(50, 1))(tower_3)

        joined_outputs = concatenate([tower_1, tower_2, tower_3], axis=-1)
        joined_outputs = Activation('relu')(joined_outputs)
        main_outputs = Flatten()(joined_outputs)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(50, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(10, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model


class ProcessorLR2DAlexNet(ProcessorLR):
    def define_cnn_model(self):
        """
        This idea failed.
        Modified AlexNet CNN model. Referred to https://engmrk.com/alexnet-implementation-using-keras/
        :return:
        """
        # # swap the IMU channel to the last axis
        # self._x_train = np.swapaxes(self._x_train, 2, 3)
        # self._x_test = np.swapaxes(self._x_test, 2, 3)

        main_input_shape = list(self._x_train.shape)
        main_input = Input((main_input_shape[1:]), name='main_input')
        kernel_regu = None

        # make the width of the kernel 3 !!!
        conv_layer = Conv2D(20, (11, 3), padding='valid', activation='relu', kernel_regularizer=kernel_regu)(main_input)
        conv_layer = Conv2D(40, (5, 3), padding='same', activation='relu', kernel_regularizer=kernel_regu)(conv_layer)
        conv_layer = Conv2D(50, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_regu)(conv_layer)
        conv_layer = Conv2D(50, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_regu)(conv_layer)
        conv_layer = Conv2D(40, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_regu)(conv_layer)
        max_pooling = MaxPool2D((20, 20), strides=(10, 10))(conv_layer)
        max_pooling = Activation('relu')(max_pooling)
        main_outputs = Flatten()(max_pooling)

        aux_input = Input(shape=(2,), name='aux_input')
        aux_joined_outputs = concatenate([main_outputs, aux_input])

        aux_joined_outputs = Dense(40, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(20, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(20, activation='relu')(aux_joined_outputs)
        aux_joined_outputs = Dense(1, activation='linear')(aux_joined_outputs)
        model = Model(inputs=[main_input, aux_input], outputs=aux_joined_outputs)
        self.model = model


class ProcessorLROnlyNormalized(ProcessorLR2DConv):
    """
    This idea failed.
    """
    def convert_input(self, input_all_list, sampling_fre):
        """
        CNN based algorithm improved
        """
        step_num = len(input_all_list)
        resample_len = self.sensor_sampling_fre
        data_clip_start, data_clip_end = int(resample_len * 0.5), int(resample_len * 0.75)
        step_input = np.zeros([step_num, data_clip_end - data_clip_start, self.channel_num])
        aux_input = np.zeros([step_num, 2])
        for i_step in range(step_num):
            acc_gyr_data = input_all_list[i_step][:, 0:self.channel_num]
            for i_channel in range(self.channel_num):
                channel_resampled = ProcessorLR.resample_channel(acc_gyr_data[:, i_channel], resample_len)
                step_input[i_step, :, i_channel] = channel_resampled[data_clip_start:data_clip_end]
                step_len = acc_gyr_data.shape[0]
                aux_input[i_step, 0] = step_len
                strike_sample_num = np.where(input_all_list[i_step][:, -1] == 1)[0]
                aux_input[i_step, 1] = strike_sample_num
        step_input = self.norm_step_input(step_input)
        aux_input = ProcessorLR.clean_aux_input(aux_input)
        return step_input, aux_input

    @staticmethod
    def norm_step_input(step_input):
        """
        Each accelerometer and gyroscope was considered as one sensor and was normalized into one single channel.
        This has to be done before channel normalization.
        :return:
        """
        step_num = step_input.shape[0]
        step_len = step_input.shape[1]
        sensor_num = int(step_input.shape[2] / 3)
        step_input_trans = np.zeros([step_num, step_len, sensor_num])
        for i_step in range(step_num):
            for i_sensor in range(sensor_num):
                step_input_trans[i_step, :, i_sensor] = np.linalg.norm(
                    step_input[i_step, :, i_sensor*3:(1+i_sensor)*3], axis=1)
        return step_input_trans



class ProcessorLROptimizerOptimizer(ProcessorLR):
    """
    Study the optimization of optimizer
    """
    pass

class ProcessorIMUIndependentTower(ProcessorLR):
    pass






