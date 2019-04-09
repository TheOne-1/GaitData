import csv
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from const import COLUMN_NAMES_HAISHENG, SEGMENT_MARKERS, FOLDER_PATH, MOCAP_SAMPLE_RATE, PLATE_SAMPLE_RATE,\
    HAISHENG_SENSOR_SAMPLE_RATE, FORCE_NAMES
from numpy.linalg import norm
import matplotlib.pyplot as plt


class ViconReader:
    def __init__(self, file):
        self._file = file
        self.marker_offset = self._find_marker_start_row()  # get the offset
        self.marker_data_all_df = self.__get_marker_processed()
        # self.plate_data = self.__get_plate_processed()

    def _find_marker_start_row(self):
        with open(self._file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            i_row = 0
            for row in reader:
                if row and (row[0] == 'Trajectories'):
                    marker_offset = i_row + 3
                    break
                i_row += 1
        return marker_offset

    def get_marker_names(self, row_num):
        with open(self._file, 'r', errors='ignore') as csvfile:
            reader = csv.reader(csvfile)
            # get the row of names
            for i_row, rows in enumerate(reader):
                if i_row == row_num:
                    the_row = rows
                    break
        names_raw = list(filter(lambda a: a != '', the_row))
        # bulid a new
        names = list()
        for name in names_raw:
            name = name.split(':')[1]
            names.append(name + '_x')
            names.append(name + '_y')
            names.append(name + '_z')
        names.insert(0, 'marker_frame')
        return names

    def _get_marker_raw(self):
        # skip the force data and a couple rows when selecting marker data
        skip_range = list(range(0, self.marker_offset + 2))
        data_raw_marker = pd.read_csv(self._file, skiprows=skip_range, header=None)
        return data_raw_marker

    def __get_marker_processed(self, cut_off_fre=12, filter_order=4):
        """
        process include filtering, adding column names

        :param cut_off_fre: cut off frequency
        :param filter_order: butterworth filter order
        :return: processed marker data
        """
        wn_marker = cut_off_fre / (MOCAP_SAMPLE_RATE / 2)
        data_raw_marker = self._get_marker_raw().values
        b_marker, a_marker = butter(filter_order, wn_marker, btype='low')
        # use low pass filter to filter marker data
        data_marker = data_raw_marker[:, 2:]  # Frame column does not need a filter
        data_marker = filtfilt(b_marker, a_marker, data_marker, axis=0)  # filtering
        data_marker = np.column_stack([data_raw_marker[:, 0], data_marker])

        column_names = self.get_marker_names(self.marker_offset-1)
        data_marker_df = pd.DataFrame(data_marker, columns=column_names)
        return data_marker_df

    def get_marker_data_processed_segment(self, segment_name):
        segment_marker_names = SEGMENT_MARKERS[segment_name]
        segment_marker_names_xyz = [name + axis for name in segment_marker_names for axis in ['_x', '_y', '_z']]
        marker_data_all = self.marker_data_all_df
        marker_data_segment = marker_data_all[segment_marker_names_xyz]
        return marker_data_segment

    def get_marker_data_one_marker(self, marker_name):
        """
        :param marker_name: target marker name
        :return:
        """
        marker_name_xyz = [marker_name + axis for axis in ['_x', '_y', '_z']]
        return self.marker_data_all_df[marker_name_xyz]

    @staticmethod
    def _get_plate_calibration(plate_1_file='plate1.csv', plate_2_file='plate2.csv'):
        # needs debugging !!!
        cop_offsets = []
        for plate_file in [plate_1_file, plate_2_file]:
            plate_reader = ViconReader(FOLDER_PATH + 'Vicon\\' + plate_file)
            data_DL = plate_reader.get_marker_data_one_marker('DL').values
            data_DR = plate_reader.get_marker_data_one_marker('DR').values
            data_ML = plate_reader.get_marker_data_one_marker('ML').values
            center_vicon = (data_DL + data_DR) / 2 + (data_DL - data_ML)
            plate_data_raw = plate_reader._get_plate_raw()
            if '1' in plate_file:
                center_plate = plate_data_raw[['Cx', 'Cy', 'Cz']].values
            elif '2' in plate_file:
                center_plate = plate_data_raw[['Cx.1', 'Cy.1', 'Cz.1']].values
            else:
                raise RuntimeError('Plate 1 or 2 not specified')
            cop_offsets.append(np.mean(center_plate - center_vicon, axis=0))
        return np.concatenate([cop_offsets[0], cop_offsets[1]])

    def _get_plate_raw(self):

        plate_data_raw = pd.read_csv(self._file, skiprows=[0, 1, 2, 4], nrows=self.marker_offset - 9).astype(float)
        # only keep useful columns
        plate_data_raw = plate_data_raw[['Frame', 'Fx', 'Fy', 'Fz', 'Cx', 'Cy', 'Cz', 'Fx.1', 'Fy.1', 'Fz.1', 'Cx.1', 'Cy.1', 'Cz.1']]
        return plate_data_raw

    def __get_plate_processed(self, wn_plate=50 / (PLATE_SAMPLE_RATE / 2), filter_order=4):
        plate_data_raw = self._get_plate_raw().values
        plate_offsets = self._get_plate_calibration()
        for channel in range(4, 7):
            plate_data_raw[:, channel] -= plate_offsets[channel-4]
        for channel in range(10, 13):
            plate_data_raw[:, channel] -= plate_offsets[channel-7]

        # set threshold to the plate force, 先设阈值再滤波
        threshold = 50
        l_total_force = norm(plate_data_raw[:, 1:4], axis=1)
        r_total_force = norm(plate_data_raw[:, 7:10], axis=1)
        # filter the force data
        plate_data = plate_data_raw[:, 1:]
        b_force_plate, a_force_plate = butter(filter_order, wn_plate, btype='low')
        plate_data = filtfilt(b_force_plate, a_force_plate, plate_data, axis=0)  # filtering
        plate_data = np.column_stack((plate_data_raw[:, 0], plate_data))  # stack the time sample
        i_row = 0
        for l_force in l_total_force:
            if l_force < threshold:
                plate_data[i_row, 1:4] = 0
            i_row += 1
        i_row = 0
        for r_force in r_total_force:
            if r_force < threshold:
                plate_data[i_row, 7:10] = 0
            i_row += 1
        plate_data_df = pd.DataFrame(plate_data, columns=FORCE_NAMES)
        return plate_data_df

    def get_plate_data_resampled(self, resample_fre=MOCAP_SAMPLE_RATE):
        """
        No interpolation included
        :return:
        """
        plate_data_df = self.__get_plate_processed()
        ratio = PLATE_SAMPLE_RATE / resample_fre
        if ratio - int(ratio) > 1e-6:    # check if ratio is an int
            raise RuntimeError('resample failed, try interpolation')
        data_len = plate_data_df.shape[0]
        force_data_range = range(0, data_len, ratio)
        return plate_data_df[force_data_range]


class HaishengSensorReader:
    def __init__(self, file):
        self._file = file

    def _get_sensor_data_raw(self):
        data_raw_haisheng = pd.read_csv(self._file)
        data_raw_haisheng.columns = COLUMN_NAMES_HAISHENG
        return data_raw_haisheng

    def get_sensor_data_processed(self, wn=20/(HAISHENG_SENSOR_SAMPLE_RATE/2), filter_order=4):
        """
        process include filtering
        :param wn: normalized frequency
        :param filter_order: butterworth filter order
        :return:
        """
        data_raw_haisheng = self._get_sensor_data_raw().values
        b_marker, a_marker = butter(filter_order, wn, btype='low')
        data_haisheng = data_raw_haisheng[:, 4:10]  # Frame column does not need a filter
        data_haisheng = filtfilt(b_marker, a_marker, data_haisheng, axis=0)  # filtering
        data_haisheng_df = pd.DataFrame(data_haisheng, columns=COLUMN_NAMES_HAISHENG[4:10])
        return data_haisheng_df

    def get_processed_channel_data(self, channel):
        data_haisheng_df = self.get_sensor_data_processed()
        if isinstance(channel, list):
            return data_haisheng_df[channel].values
        elif isinstance(channel, str):
            return data_haisheng_df[[channel]].values.reshape(-1)
        else:
            raise RuntimeError('Wrong channel type')

    def get_gyr_norm(self):
        data_haisheng_df = self.get_sensor_data_processed()
        gyr_data = data_haisheng_df[['gyr_x', 'gyr_y', 'gyr_z']]
        return norm(gyr_data, axis=1)

    @staticmethod
    def sync_with_vicon(vicon_gyr, sensor_gyr, check=False):
        correlation = np.correlate(vicon_gyr, sensor_gyr, 'valid')
        vicon_data_len = len(vicon_gyr)
        sensor_data_len = len(sensor_gyr)
        # normalize correlation data
        for i_sample in range(sensor_data_len - vicon_data_len + 1):
            norm_range = range(i_sample, i_sample+vicon_data_len)
            correlation[i_sample] = correlation[i_sample] / norm(sensor_gyr[norm_range])

        if check:
            plt.plot(correlation)
            print(np.argmax(correlation) + vicon_data_len)
            # plt.show()
        diff = len(sensor_gyr) - np.argmax(correlation) - 1
        return diff

