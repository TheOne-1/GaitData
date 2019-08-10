import csv
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from const import SEGMENT_MARKERS, PLATE_SAMPLE_RATE, HAISHENG_SENSOR_SAMPLE_RATE, FORCE_NAMES, MOCAP_SAMPLE_RATE, \
    COP_DIFFERENCE


class ViconReader:
    def __init__(self, file):
        self._file = file
        self.marker_start_row = self._find_marker_start_row()  # get the offset
        self.marker_data_processed_df = self.__get_marker_processed()

    def _find_marker_start_row(self):
        """
        For the csv file exported by Vicon Nexus, this function find the start row of the marker
        :return: int, marker start row
        """
        with open(self._file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            i_row = 0
            for row in reader:
                if row and (row[0] == 'Trajectories'):
                    marker_start_row = i_row + 3
                    break
                i_row += 1
        return marker_start_row

    def _get_marker_names(self, row_num):
        """
        This function automatically find all the marker names
        :param row_num: int, the row number of marker
        :return: list, list of marker names
        """
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
        skip_range = list(range(0, self.marker_start_row + 2))
        data_raw_marker = pd.read_csv(self._file, skiprows=skip_range, header=None)
        return data_raw_marker

    def __get_marker_processed(self, cut_off_fre=12, filter_order=4):
        """
        process include filtering, adding column names
        :param cut_off_fre: int, cut off frequency
        :param filter_order: int, butterworth filter order
        :return: Dataframe, processed marker data
        """
        wn_marker = cut_off_fre / (MOCAP_SAMPLE_RATE / 2)
        data_raw_marker = self._get_marker_raw().values
        b_marker, a_marker = butter(filter_order, wn_marker, btype='low')
        # use low pass filter to filter marker data
        data_marker = data_raw_marker[:, 2:]  # Frame column does not need a filter
        data_marker = filtfilt(b_marker, a_marker, data_marker, axis=0)  # filtering
        data_marker = np.column_stack([data_raw_marker[:, 0], data_marker])

        column_names = self._get_marker_names(self.marker_start_row - 1)
        data_marker_df = pd.DataFrame(data_marker, columns=column_names)
        return data_marker_df

    def get_marker_data_processed_segment(self, segment_name):
        """
        Return the marker dataframe of one segment
        :param segment_name: str, name of the target segment
        :return: dataframe, marker data of this segment
        """
        segment_marker_names = SEGMENT_MARKERS[segment_name]
        segment_marker_names_xyz = [name + axis for name in segment_marker_names for axis in ['_x', '_y', '_z']]
        marker_data_all = self.marker_data_processed_df
        marker_data_segment_df = marker_data_all[segment_marker_names_xyz]
        return marker_data_segment_df

    def get_marker_data_one_marker(self, marker_name):
        """
        :param marker_name: target marker name
        :return: dataframe,
        """
        marker_name_xyz = [marker_name + axis for axis in ['_x', '_y', '_z']]
        return self.marker_data_processed_df[marker_name_xyz]

    @staticmethod
    def _get_plate_calibration(file):
        """
        To get force plate calibration.
        A ViconReader object is implemented using force plate 1 data.
        :param file: str, any file that in the same folder as plate1.csv
        :return: numpy.ndarry
        """
        name_index = file.rfind('\\')
        plate_file = file[:name_index] + '\\plate1.csv'
        plate_reader = ViconReader(plate_file)
        data_DL = plate_reader.get_marker_data_one_marker('DL').values
        data_DR = plate_reader.get_marker_data_one_marker('DR').values
        data_ML = plate_reader.get_marker_data_one_marker('ML').values
        center_vicon = (data_DL + data_DR) / 2 + (data_DL - data_ML)
        plate_data_raw = plate_reader._get_plate_data_raw_resampled()
        center_plate = plate_data_raw[['Cx', 'Cy', 'Cz']].values
        cop_offset = np.mean(center_plate - center_vicon, axis=0)
        # cop_offset += COP_DIFFERENCE
        return cop_offset

    def _get_plate_raw(self):
        plate_data_raw = pd.read_csv(self._file, skiprows=[0, 1, 2, 4], nrows=self.marker_start_row - 9).astype(float)
        # only keep useful columns
        plate_data_raw = plate_data_raw[
            ['Frame', 'Fx', 'Fy', 'Fz', 'Cx', 'Cy', 'Cz', 'Fx.1', 'Fy.1', 'Fz.1', 'Cx.1', 'Cy.1', 'Cz.1']]
        return plate_data_raw

    def get_plate_processed(self, cut_off_fre=50, filter_order=4):
        """
        Process include COP calibration and filtering.
        :param cut_off_fre: int, cut off frequency of the butterworth low pass filter
        :param filter_order: int, butterworth filter order
        :return: dataframe, force and COP data
        """
        plate_data_raw = self._get_plate_raw().values
        # calibrate COP differences between force plate and vicon
        plate_offsets = self._get_plate_calibration(self._file)
        for channel in range(4, 7):  # Minus the COP offset of the first plate
            plate_data_raw[:, channel] = plate_data_raw[:, channel] - plate_offsets[channel - 4]

        # filter the force data
        plate_data = plate_data_raw[:, 1:]
        wn_plate = cut_off_fre / (PLATE_SAMPLE_RATE / 2)
        b_force_plate, a_force_plate = butter(filter_order, wn_plate, btype='low')
        plate_data_filtered = filtfilt(b_force_plate, a_force_plate, plate_data, axis=0)  # filtering

        plate_data_filtered = np.column_stack((plate_data_raw[:, 0], plate_data_filtered))  # stack the time sample
        plate_data_df = pd.DataFrame(plate_data_filtered, columns=FORCE_NAMES)
        return plate_data_df

    def _get_plate_data_raw_resampled(self, resample_fre=MOCAP_SAMPLE_RATE):
        """
        The returned data is unfiltered. No interpolation included
        :param resample_fre: int
        :return: dataframe, force and COP data
        """
        plate_data_df = self._get_plate_raw()
        ratio = PLATE_SAMPLE_RATE / resample_fre
        if ratio - int(ratio) > 1e-6:  # check if ratio is an int
            raise RuntimeError('resample failed, try interpolation')
        data_len = plate_data_df.shape[0]
        force_data_range = range(0, data_len, int(ratio))
        return plate_data_df.loc[force_data_range]

    def get_plate_data_resampled(self, resample_fre=MOCAP_SAMPLE_RATE):
        """
        The returned data is filtered. No interpolation included.
        :return: dataframe, force and COP data
        """
        plate_data_df = self.get_plate_processed()
        ratio = PLATE_SAMPLE_RATE / resample_fre
        if ratio - int(ratio) > 1e-6:  # check if ratio is an int
            raise ValueError('resample failed, try interpolation')
        data_len = plate_data_df.shape[0]
        force_data_range = range(0, data_len, int(ratio))
        resampled_plate_data_df = plate_data_df.loc[force_data_range].copy()
        resampled_plate_data_df = resampled_plate_data_df.reset_index(drop=True)
        return resampled_plate_data_df

    @staticmethod
    def resample_data(ori_data, target_fre=HAISHENG_SENSOR_SAMPLE_RATE, ori_fre=MOCAP_SAMPLE_RATE):
        """
        The returned data is filtered. No interpolation included.
        :param ori_data: Dataframe or ndarray
        :param target_fre:
        :param ori_fre:
        :return:
        """
        ratio = ori_fre / target_fre
        if ratio - int(ratio) > 1e-6:  # check if ratio is an int
            raise ValueError('resample failed, try interpolation')
        data_len = ori_data.shape[0]
        data_range = range(0, data_len, int(ratio))
        if isinstance(ori_data, pd.DataFrame):
            resampled_data_df = ori_data.loc[data_range]
            return resampled_data_df.reset_index(drop=True)
        else:
            if len(ori_data.shape) == 1:
                return ori_data[data_range]
            elif len(ori_data.shape) == 2:
                return ori_data[data_range, :]
            else:
                raise ValueError('Wrong dimension')

    def get_vicon_all_processed_df(self):
        plate_df = self.get_plate_data_resampled()
        all_data_df = pd.concat([plate_df, self.marker_data_processed_df], axis=1)
        return all_data_df










