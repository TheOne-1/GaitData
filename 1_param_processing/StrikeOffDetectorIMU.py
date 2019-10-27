import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, lfilter, firwin
from const import FILTER_WIN_LEN, TRIAL_START_BUFFER
from scipy import signal


class StrikeOffDetectorIMU:
    """
    This class detect strike, off event via IMU data
    """

    def __init__(self, trial_name, gait_data_df, param_data_df, IMU_location, sampling_fre):
        self._trial_name = trial_name
        self._gait_data_df = gait_data_df
        self._param_data_df = param_data_df
        self._IMU_location = IMU_location
        self._sampling_fre = sampling_fre

    @staticmethod
    def check_true_values(gait_event, step_num, diff_ratio=0.15):
        """
        Check strike/off via step length. Any strike that larger or lower than certain ratio will be abandoned
        :param gait_event:
        :param step_num:
        :param diff_ratio:
        :return:
        """
        step_lens = np.zeros([step_num])
        for i_step in range(step_num):
            step_lens[i_step] = gait_event[i_step + 1] - gait_event[i_step]
        step_len_mean = np.mean(step_lens)
        max_len = step_len_mean * (1 + diff_ratio)
        min_len = step_len_mean * (1 - diff_ratio)
        strikes_checked = []
        for i_step in range(step_num):
            if min_len < step_lens[i_step] < max_len:
                strikes_checked.append(gait_event[i_step])
        step_num_checked = len(strikes_checked)
        return np.array(strikes_checked), step_num_checked

    def get_IMU_data(self, acc=True, gyr=False, mag=False):
        column_names = []
        if acc:
            column_names += [self._IMU_location + '_acc_' + axis for axis in ['x', 'y', 'z']]
        if gyr:
            column_names += [self._IMU_location + '_gyr_' + axis for axis in ['x', 'y', 'z']]
        if mag:
            column_names += [self._IMU_location + '_mag_' + axis for axis in ['x', 'y', 'z']]
        return self._gait_data_df[column_names]

    def true_esti_diff(self, estimated_event_indexes, event_name):
        """
        Compare the strike/off detection result between force plate and IMU
        :return:
        """
        side = self._IMU_location[0]
        true_event = self._param_data_df[side + '_' + event_name]
        true_event_indexes = np.where(true_event == 1)[0]
        true_len = true_event_indexes.shape[0]
        estimated_len = len(estimated_event_indexes)
        diffs = []
        for i_true in range(true_len):
            for i_esti in range(estimated_len):
                # the IMU detection result is considered valid as long as the difference is lower than 15
                if abs(true_event_indexes[i_true] - estimated_event_indexes[i_esti]) < 20:
                    diffs.append(true_event_indexes[i_true] - estimated_event_indexes[i_esti])
        diff_mean, diff_std, diff_in_num = np.mean(diffs), np.std(diffs), true_len - estimated_len
        print(self._IMU_location + ' IMU ' + event_name + ' detection result: difference mean = ' + str(diff_mean) +
              ', difference std = ' + str(diff_std) + ', number difference = ' + str(diff_in_num))
        return diff_mean, diff_std, diff_in_num

    @staticmethod
    def data_filt(data, cut_off_fre, sampling_fre, filter_order=4):
        fre = cut_off_fre / (sampling_fre / 2)
        b, a = butter(filter_order, fre, 'lowpass')
        if len(data.shape) == 1:
            data_filt = filtfilt(b, a, data)
        else:
            data_filt = filtfilt(b, a, data, axis=0)
        return data_filt

    def std_check(self, check_win_len=200, sliding_win_len=10):
        """
        This function is used for walking running classification.
        :param check_win_len:
        :param sliding_win_len:
        :return:
        """
        acc_data = self.get_IMU_data(acc=True, gyr=False).values
        acc_z = acc_data[:, 0]
        data_len = acc_z.shape[0]
        result = np.zeros([int(np.ceil((data_len - check_win_len) / sliding_win_len))])
        for i_sample in range(0, data_len - check_win_len, sliding_win_len):
            result[int(i_sample / sliding_win_len)] = np.std(acc_z[i_sample:i_sample + check_win_len])
        return result

    def show_IMU_data_and_strike_off(self, estimated_strike_indexes, estimated_off_indexes):
        """
        This function is used for giving ideas about
        :return:
        """
        side = self._IMU_location[0]
        true_strikes = self._param_data_df[side + '_strikes']
        true_strike_indexes = np.where(true_strikes == 1)[0]
        true_offs = self._param_data_df[side + '_offs']
        true_off_indexes = np.where(true_offs == 1)[0]

        # acc_data = self.get_IMU_data(acc=True, gyr=False).values
        acc_gyr_data = self.get_IMU_data(acc=True, gyr=True).values
        acc_plus = acc_gyr_data[:, 1] + acc_gyr_data[:, 2]
        gyr_plus = acc_gyr_data[:, 4] - acc_gyr_data[:, 5]
        acc_plus = self.data_filt(acc_plus, cut_off_fre=6, sampling_fre=200)
        gyr_plus = self.data_filt(gyr_plus, cut_off_fre=6, sampling_fre=200)

        plt.figure()
        plt.title(self._trial_name + '   ' + self._IMU_location + '   acc_z')
        plt.plot(acc_plus)
        strike_plt_handle = plt.plot(true_strike_indexes, acc_plus[true_strike_indexes], 'g*')
        off_plt_handle = plt.plot(true_off_indexes, acc_plus[true_off_indexes], 'gx')
        strike_plt_handle_esti = plt.plot(estimated_strike_indexes, acc_plus[estimated_strike_indexes], 'r*')
        off_plt_handle_esti = plt.plot(estimated_off_indexes, acc_plus[estimated_off_indexes], 'rx')
        plt.grid()
        plt.legend([strike_plt_handle[0], off_plt_handle[0], strike_plt_handle_esti[0], off_plt_handle_esti[0]],
                   ['true_strikes', 'true_offs', 'estimated_strikes', 'estimated_offs'])

        plt.figure()
        plt.title(self._trial_name + '   ' + self._IMU_location + '   gyr_x')
        plt.plot(gyr_plus)
        strike_plt_handle = plt.plot(true_strike_indexes, gyr_plus[true_strike_indexes], 'g*')
        off_plt_handle = plt.plot(true_off_indexes, gyr_plus[true_off_indexes], 'gx')
        strike_plt_handle_esti = plt.plot(estimated_strike_indexes, gyr_plus[estimated_strike_indexes], 'r*')
        off_plt_handle_esti = plt.plot(estimated_off_indexes, gyr_plus[estimated_off_indexes], 'rx')
        plt.grid()
        plt.legend([strike_plt_handle[0], off_plt_handle[0], strike_plt_handle_esti[0], off_plt_handle_esti[0]],
                   ['true_strikes', 'true_offs', 'estimated_strikes', 'estimated_offs'])


class StrikeOffDetectorFootIMU(StrikeOffDetectorIMU):

    def get_jogging_strike_off(self, strike_delay, off_delay):
        strike_acc_z_thd = 3.5  # threshold of the maximum peak of acc z
        strike_acc_prominence = 5
        off_gyr_thd = 4  # threshold the minimum peak of medio-lateral heel strike
        off_gyr_prominence = 2

        skip_len_after_strike = int(15 * self._sampling_fre / 100)  # find the off at least 150 ms after strike
        skip_len_after_off = int(20 * self._sampling_fre / 100)  # find the strike at least 200 ms after off

        acc_data = self.get_IMU_data(acc=True, gyr=False).values
        acc_z_unfilt = acc_data[:, 2]
        acc_z = StrikeOffDetectorIMU.data_filt(acc_z_unfilt, 10, self._sampling_fre)

        gyr_data = self.get_IMU_data(acc=False, gyr=True).values
        gyr_x_unfilt = gyr_data[:, 0]
        gyr_x = StrikeOffDetectorIMU.data_filt(gyr_x_unfilt, 10, self._sampling_fre)

        # step 0, find the falling before the first strike
        win_len = 4
        sliding_len = 2
        data_len = acc_z.shape[0]
        start_sample = 0
        for i_sample in range(0, data_len, sliding_len):
            if (abs(acc_z[i_sample:i_sample + win_len]) > strike_acc_z_thd).all():
                start_sample = i_sample
                break
        is_swing_phase = True

        # step 1, find the following phases
        estimated_strike_indexes, estimated_off_indexes = [], []
        i_sample = start_sample
        max_peak_window_len = 500  # find the peak within 500 samples

        try:
            while i_sample < data_len:
                peak_window_len = min(max_peak_window_len, data_len - i_sample)

                # during swing phase find the strike
                if is_swing_phase:
                    peaks, _ = find_peaks(-acc_z[i_sample:i_sample + peak_window_len],
                                          height=strike_acc_z_thd, prominence=strike_acc_prominence)
                    the_peak_sample = peaks[0] + i_sample + strike_delay
                    estimated_strike_indexes.append(the_peak_sample)
                    i_sample = peaks[0] + i_sample + skip_len_after_strike
                    is_swing_phase = False

                # during stance phase, find the off
                else:
                    peaks, _ = find_peaks(-gyr_x[i_sample:i_sample + peak_window_len],
                                          height=off_gyr_thd, prominence=off_gyr_prominence)
                    the_peak_sample = peaks[0] + i_sample + off_delay
                    estimated_off_indexes.append(the_peak_sample)
                    i_sample = peaks[0] + i_sample + skip_len_after_off
                    is_swing_phase = True

        except IndexError:
            # in case of no peak was found
            if len(peaks) == 0:  # in case of no peak was found
                if peak_window_len == max_peak_window_len:
                    raise RuntimeError('Acc peak not found.')
            # in case of i_sample out of range
            if i_sample + peak_window_len > data_len:
                pass
        if estimated_off_indexes[-1] >= data_len:
            estimated_off_indexes.pop(-1)
        if estimated_strike_indexes[-1] >= data_len:
            estimated_strike_indexes.pop(-1)
        return estimated_strike_indexes, estimated_off_indexes


class StrikeOffDetectorShankIMU(StrikeOffDetectorIMU):

    def get_jogging_strike_off(self, strike_delay, off_delay):
        acc_gyr_data = self.get_IMU_data(acc=True, gyr=True).values
        acc_plus_unfilt = acc_gyr_data[:, 1] + acc_gyr_data[:, 2]
        acc_plus = StrikeOffDetectorIMU.data_filt(acc_plus_unfilt, 6, self._sampling_fre)

        gyr_plus_unfilt = acc_gyr_data[:, 4] - acc_gyr_data[:, 5]
        gyr_plus = StrikeOffDetectorIMU.data_filt(gyr_plus_unfilt, 6, self._sampling_fre)

        trial_start_buffer_sample_num = TRIAL_START_BUFFER * self._sampling_fre

        off_acc_thd = 0  # threshold of the maximum peak of acc plus
        off_acc_prominence = 10

        strike_list, off_list = [], []

        data_len = acc_plus.shape[0]
        # step 0, find the first off
        peaks, _ = find_peaks(
            acc_plus[trial_start_buffer_sample_num:trial_start_buffer_sample_num + self._sampling_fre * 2],
            height=off_acc_thd, prominence=off_acc_prominence)
        try:
            last_off = peaks[-1] + trial_start_buffer_sample_num + off_delay
            off_list.append(last_off)
            i_sample = last_off + 1
        except IndexError:
            plt.figure()
            plt.plot(acc_plus[trial_start_buffer_sample_num:trial_start_buffer_sample_num + self._sampling_fre * 2])
            plt.show()
            raise IndexError('Gyr peak not found')

        gyr_peak_win_len = int(30 * self._sampling_fre / 100)
        acc_peak_window_len = int(130 * self._sampling_fre / 100)
        skip_len_after_strike = 1      # find the off at least 150 ms after strike
        skip_len_after_off = 1      # find the strike at least 200 ms after off

        is_swing_phase = True
        # step 1, find the following phases
        try:
            while i_sample < data_len:

                # during swing phase find the strike
                if is_swing_phase:
                    # go to the negative region
                    while gyr_plus[i_sample] > -3:
                        i_sample += 1
                    # got to the positive region
                    while gyr_plus[i_sample] < 0:
                        i_sample += 1

                    peaks, _ = find_peaks(gyr_plus[i_sample:i_sample + gyr_peak_win_len])
                    the_peak_sample = peaks[0] + i_sample + strike_delay
                    strike_list.append(the_peak_sample)
                    i_sample = peaks[0] + i_sample + skip_len_after_strike
                    is_swing_phase = False

                # during stance phase, find the off
                else:

                    peaks, _ = find_peaks(acc_plus[i_sample:i_sample + acc_peak_window_len], height=off_acc_thd,
                                          prominence=off_acc_prominence)
                    the_peak_sample = peaks[0] + i_sample + off_delay
                    off_list.append(the_peak_sample)
                    i_sample = peaks[0] + i_sample + skip_len_after_off
                    is_swing_phase = True

        except IndexError:
            # in case of i_sample out of range
            if i_sample + acc_peak_window_len > data_len:
                pass
            # in case of no peak was found
            elif len(peaks) == 0:  # in case of no peak was found
                raise RuntimeError('Acc peak not found.')
        if off_list[-1] >= data_len:
            off_list.pop(-1)
        if strike_list[-1] >= data_len:
            strike_list.pop(-1)

        return strike_list, off_list


class StrikeOffDetectorIMUFilter(StrikeOffDetectorIMU):
    """
    lfilter rather than filtfilt is used in this class
    """

    @staticmethod
    def data_filt(data, cut_off_fre, sampling_fre, filter_order=4):
        wn = cut_off_fre / sampling_fre
        b = firwin(FILTER_WIN_LEN, wn)
        if len(data.shape) == 1:
            data_filt = lfilter(b, 1, data)
        else:
            data_filt = lfilter(b, 1, data, axis=0)
        return data_filt

    def true_esti_diff(self, estimated_event_indexes, event_name):
        """
        Compare the strike/off detection result between force plate and IMU
        :return:
        """
        side = self._IMU_location[0]
        true_event = self._param_data_df[side + '_' + event_name]
        filter_delay = int(FILTER_WIN_LEN / 2)
        true_event_indexes = np.where(true_event == 1)[0][:-1] + filter_delay
        true_len = true_event_indexes.shape[0]
        estimated_len = len(estimated_event_indexes)
        diffs = []
        for i_true in range(true_len):
            for i_esti in range(estimated_len):
                # the IMU detection result is considered valid as long as the difference is lower than 15
                if abs(true_event_indexes[i_true] - estimated_event_indexes[i_esti]) < 20:
                    diffs.append(true_event_indexes[i_true] - estimated_event_indexes[i_esti])
        diff_mean, diff_std, diff_in_num = np.mean(diffs), np.std(diffs), true_len - estimated_len
        print(self._IMU_location + ' IMU ' + event_name + ' detection result: difference mean = ' + str(diff_mean) +
              ', difference std = ' + str(diff_std) + ', number difference = ' + str(diff_in_num))
        return diff_in_num

    def find_peak_max(self, data_clip, height, width=None, prominence=None):
        """
        find the maximum peak
        :return:
        """
        peaks, properties = signal.find_peaks(data_clip, width=width, height=height, prominence=prominence)
        peak_heights = properties['peak_heights']
        max_index = np.argmax(peak_heights)
        return peaks[max_index]

    def get_jogging_strike_off(self, strike_delay, off_delay, cut_off_fre_strike_off=10):
        strike_acc_width = 10
        strike_acc_prominence = 6
        strike_acc_height = -15
        off_gyr_thd = 4  # threshold the minimum peak of medio-lateral heel strike
        off_gyr_prominence = 0.1

        # strike_delay, off_delay = 0, 0  # set as zero for debugging

        acc_data = self.get_IMU_data(acc=True, gyr=False).values
        acc_z_unfilt = -acc_data[:, 2]
        acc_z_filtered = self.data_filt(acc_z_unfilt, cut_off_fre_strike_off, self._sampling_fre)

        gyr_data = self.get_IMU_data(acc=False, gyr=True).values
        gyr_x_unfilt = -gyr_data[:, 0]
        gyr_x_filtered = self.data_filt(gyr_x_unfilt, cut_off_fre_strike_off, self._sampling_fre)

        data_len = acc_data.shape[0]
        strike_list, off_list = [], []
        trial_start_buffer_sample_num = (TRIAL_START_BUFFER + 1) * self._sampling_fre

        # find the first off
        peaks, _ = signal.find_peaks(gyr_x_filtered[:trial_start_buffer_sample_num], height=off_gyr_thd,
                                     prominence=off_gyr_prominence)
        try:
            last_off = peaks[-1] + off_delay
        except IndexError:
            plt.figure()
            plt.plot(gyr_x_filtered[:trial_start_buffer_sample_num])
            plt.show()
            raise IndexError('Gyr peak not found')

        # find strikes and offs (with filter delays)
        check_win_len = self._sampling_fre
        for i_sample in range(trial_start_buffer_sample_num + 1, data_len):
            if i_sample - last_off > check_win_len:
                try:
                    acc_peak = self.find_peak_max(acc_z_filtered[last_off:i_sample - int(check_win_len / 4)],
                                                  width=strike_acc_width,
                                                  prominence=strike_acc_prominence, height=strike_acc_height)
                    gyr_peak = self.find_peak_max(gyr_x_filtered[last_off:i_sample],
                                                  height=off_gyr_thd, prominence=off_gyr_prominence)
                    strike_list.append(acc_peak + last_off + strike_delay)
                    off_list.append(gyr_peak + last_off + off_delay)
                    last_off = off_list[-1]
                except ValueError as e:
                    plt.figure()
                    plt.plot(acc_z_filtered[last_off:i_sample - int(check_win_len / 4)])
                    plt.plot(gyr_x_filtered[last_off:i_sample])
                    plt.grid()
                    plt.show()
                    last_off = last_off + int(self._sampling_fre * 0.7)  # skip this step
        return strike_list, off_list

    def show_IMU_data_and_strike_off(self, estimated_strike_indexes, estimated_off_indexes):
        """
        This function is used for giving ideas about
        :return:
        """
        side = self._IMU_location[0]
        true_strikes = self._param_data_df[side + '_strikes']
        filter_delay = int(FILTER_WIN_LEN / 2)
        true_strike_indexes = np.where(true_strikes == 1)[0][:-1] + filter_delay  # Add the filter delay
        true_offs = self._param_data_df[side + '_offs']
        true_off_indexes = np.where(true_offs == 1)[0][:-1] + filter_delay  # Add the filter delay

        acc_data = self.get_IMU_data(acc=True, gyr=False).values
        acc_z = -acc_data[:, 2]
        acc_z = self.data_filt(acc_z, 10, self._sampling_fre)
        plt.figure()
        plt.title(self._trial_name + '   ' + self._IMU_location + '   acc_z')
        plt.plot(acc_z)
        strike_plt_handle = plt.plot(true_strike_indexes, acc_z[true_strike_indexes], 'g*')
        off_plt_handle = plt.plot(true_off_indexes, acc_z[true_off_indexes], 'gx')
        strike_plt_handle_esti = plt.plot(estimated_strike_indexes, acc_z[estimated_strike_indexes], 'r*')
        off_plt_handle_esti = plt.plot(estimated_off_indexes, acc_z[estimated_off_indexes], 'rx')
        plt.grid()
        plt.legend([strike_plt_handle[0], off_plt_handle[0], strike_plt_handle_esti[0], off_plt_handle_esti[0]],
                   ['true_strikes', 'true_offs', 'estimated_strikes', 'estimated_offs'])

        gyr_data = self.get_IMU_data(acc=False, gyr=True).values
        gyr_x = -gyr_data[:, 0]
        gyr_x = self.data_filt(gyr_x, 10, self._sampling_fre)
        plt.figure()
        plt.title(self._trial_name + '   ' + self._IMU_location + '   gyr_x')
        plt.plot(gyr_x)
        strike_plt_handle = plt.plot(true_strike_indexes, gyr_x[true_strike_indexes], 'g*')
        off_plt_handle = plt.plot(true_off_indexes, gyr_x[true_off_indexes], 'gx')
        strike_plt_handle_esti = plt.plot(estimated_strike_indexes, gyr_x[estimated_strike_indexes], 'r*')
        off_plt_handle_esti = plt.plot(estimated_off_indexes, gyr_x[estimated_off_indexes], 'rx')
        plt.grid()
        plt.legend([strike_plt_handle[0], off_plt_handle[0], strike_plt_handle_esti[0], off_plt_handle_esti[0]],
                   ['true_strikes', 'true_offs', 'estimated_strikes', 'estimated_offs'])
