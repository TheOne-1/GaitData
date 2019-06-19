from numpy.linalg import norm
import abc
from scipy.signal import butter, filtfilt


class IMUSensorReader:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        All three attributes (data_raw_df, data_processed_df, and _sampling_rate) should be initialized in the
         subclass of this abstract class
        """
        self.data_raw_df = None
        self.data_processed_df = None
        self._sampling_rate = None

    @abc.abstractmethod
    def _get_raw_data(self):
        """
        get raw IMU data from either Xsens file or Haisheng's sensor file. This method has to be overrided.
        """
        pass

    def _get_channel_data_raw(self, channel):
        """
        :param channel: str or list.
        For str, acceptable names are: 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z'
        For list, each element should be one of the str specified above.
        :return:
        """
        data_raw = self.data_raw_df
        if isinstance(channel, list):
            return data_raw[channel]
        elif isinstance(channel, str):
            return data_raw[[channel]]
        else:
            raise ValueError('Wrong channel type')

    def get_sensor_data_processed(self):
        """
        :return: Return the whole processed data dataframe
        """
        return self.data_processed_df

    def get_channel_data_processed(self, channel):
        """
        :param channel: str or list
        For str, acceptable names are: 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z'
        For list, each element should be one of the str specified above.
        :return:
        """
        if isinstance(channel, list):
            return self.data_processed_df[channel]
        elif isinstance(channel, str):
            return self.data_processed_df[[channel]]
        else:
            raise ValueError('Wrong channel type')

    def get_normalized_gyr(self, processed=True, filtering=True):
        """
        :param processed: boolean, true by default
        :return: 1-d array
        """
        if processed:
            gyr_data = self.data_processed_df[['gyr_x', 'gyr_y', 'gyr_z']]
        else:
            gyr_data = self.data_raw_df[['gyr_x', 'gyr_y', 'gyr_z']]

        gyr_norm = norm(gyr_data, axis=1)
        if filtering:
            cut_off_fre = 20
            filter_order = 4
            wn = cut_off_fre / self._sampling_rate
            b, a = butter(filter_order, wn, 'lowpass')
            gyr_norm = filtfilt(b, a, gyr_norm)
        return gyr_norm

    def get_normalized_acc(self, processed):
        """
        :param processed: boolean, true by default
        :return: 1-d array
        """
        if processed:
            acc_data = self.data_processed_df[['acc_x', 'acc_y', 'acc_z']]
        else:
            acc_data = self.data_raw_df[['acc_x', 'acc_y', 'acc_z']]
        return norm(acc_data, axis=1)

