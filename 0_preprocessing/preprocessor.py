import csv
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from const import COLUMN_NAMES_HAISHENG, SEGMENT_MARKERS, FOLDER_PATH, PLATE_SAMPLE_RATE,\
    HAISHENG_SENSOR_SAMPLE_RATE, FORCE_NAMES, DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE, DATA_COLUMNS_IMU
from numpy.linalg import norm
import matplotlib.pyplot as plt
import xsensdeviceapi.xsensdeviceapi_py36_64 as xda
from threading import Lock
import time
import abc
import os
import xlrd
from shutil import copyfile


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

    def _get_sensor_data_processed(self, cut_off_fre=20, filter_order=4):
        """
        This function is invoked during initialization. Please use self.data_processed_df to get data
        process include filtering
        :param cut_off_fre: int, cut-off frequency
        :param filter_order: int, butterworth filter order
        :return:
        """
        wn = cut_off_fre / (self._sampling_rate / 2)
        data_raw = self.data_raw_df[DATA_COLUMNS_IMU]
        b_marker, a_marker = butter(filter_order, wn, btype='low')
        data_processed = data_raw.values  # Frame column does not need a filter
        data_processed = filtfilt(b_marker, a_marker, data_processed, axis=0)  # filtering
        data_processed_df = pd.DataFrame(data_processed, columns=DATA_COLUMNS_IMU)
        return data_processed_df

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


class XsensReader(IMUSensorReader):
    def __init__(self, file, cut_off_fre=20, filter_order=4):
        super().__init__()
        self._file = file
        self._sampling_rate = MOCAP_SAMPLE_RATE
        device, callback, control = self.__initialize_device()
        self.data_raw_df = self._get_raw_data(device, callback, control)
        self.data_processed_df = self._get_sensor_data_processed(cut_off_fre, filter_order)

    def __initialize_device(self):
        control = xda.XsControl_construct()
        assert(control is not 0)        # check if control is 0; raise Exception if not.
        if not control.openLogFile(self._file):
            raise RuntimeError('Failed to open log file. Aborting.')

        deviceIdArray = control.mainDeviceIds()
        for i in range(deviceIdArray.size()):
            if deviceIdArray[i].isMti() or deviceIdArray[i].isMtig():
                mtDevice = deviceIdArray[i]
                break

        if not mtDevice:
            raise RuntimeError('No MTi device found. Aborting.')

        # Get the device object
        device = control.device(mtDevice)
        assert(device is not 0)        # check if device is found; raise Exception if not.
        print('Device: %s, with ID: %s found in file' % (device.productCode(), device.deviceId().toXsString()))

        # Create and attach callback handler to device
        callback = XdaCallback()
        device.addCallbackHandler(callback)

        # By default XDA does not retain data for reading it back.
        # By enabling this option XDA keeps the buffered data in a cache so it can be accessed
        # through XsDevice::getDataPacketByIndex or XsDevice::takeFirstDataPacketInQueue
        device.setOptions(xda.XSO_RetainBufferedData, xda.XSO_None)

        # Load the log file and wait until it is loaded
        # Wait for logfile to be fully loaded, there are three ways to do this:
        # - callback: Demonstrated here, which has loading progress information
        # - waitForLoadLogFileDone: Blocking function, returning when file is loaded
        # - isLoadLogFileInProgress: Query function, used to query the device if the loading is done
        #
        # The callback option is used here.
        print('Loading the file ...')
        device.loadLogFile()
        while callback.progress() != 100:
            time.sleep(0.01)
        print('File is fully loaded')
        return device, callback, control

    def _get_raw_data(self, device, callback, control):
        # Get total number of samples
        packetCount = device.getDataPacketCount()
        # Export the data
        print('Reading data into memory ...')
        # data_df = pd.DataFrame()
        data_mat = np.zeros([packetCount, 13])
        for index in range(packetCount):
            # Retrieve a packet
            packet = device.getDataPacketByIndex(index)
            acc = packet.calibratedAcceleration()
            gyr = packet.calibratedGyroscopeData()
            mag = packet.calibratedMagneticField()
            if len(mag) != 3:
                data_mat[index, 0:6] = np.concatenate([acc, gyr])
            else:
                data_mat[index, 0:9] = np.concatenate([acc, gyr, mag])
            quaternion = packet.orientationQuaternion()
            data_mat[index, 9:13] = quaternion
        data_raw_df = pd.DataFrame(data_mat)
        data_raw_df.columns = DATA_COLUMNS_XSENS
        device.removeCallbackHandler(callback)
        control.close()
        return data_raw_df


class XdaCallback(xda.XsCallback):
    """
    This class is created by Xsens B.V.
    """
    def __init__(self):
        xda.XsCallback.__init__(self)
        self.m_progress = 0
        self.m_lock = Lock()

    def progress(self):
        return self.m_progress

    def onProgressUpdated(self, dev, current, total, identifier):
        self.m_lock.acquire()
        self.m_progress = current
        self.m_lock.release()


class ViconReader:
    def __init__(self, file):
        self._file = file
        self.marker_start_row = self._find_marker_start_row()  # get the offset
        self.marker_data_all_df = self.__get_marker_processed()

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
        marker_data_all = self.marker_data_all_df
        marker_data_segment_df = marker_data_all[segment_marker_names_xyz]
        return marker_data_segment_df

    def get_marker_data_one_marker(self, marker_name):
        """
        :param marker_name: target marker name
        :return: dataframe,
        """
        marker_name_xyz = [marker_name + axis for axis in ['_x', '_y', '_z']]
        return self.marker_data_all_df[marker_name_xyz]

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
        cop_offset += np.array([279.4, 784, 0])     # reset coordinate difference
        return cop_offset

    def _get_plate_raw(self):
        plate_data_raw = pd.read_csv(self._file, skiprows=[0, 1, 2, 4], nrows=self.marker_start_row - 9).astype(float)
        # only keep useful columns
        plate_data_raw = plate_data_raw[['Frame', 'Fx', 'Fy', 'Fz', 'Cx', 'Cy', 'Cz', 'Fx.1', 'Fy.1', 'Fz.1', 'Cx.1', 'Cy.1', 'Cz.1']]
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
        for channel in range(4, 7):     # Minus the COP offset of the first plate
            plate_data_raw[:, channel] -= plate_offsets[channel-4]

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
        if ratio - int(ratio) > 1e-6:    # check if ratio is an int
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
        if ratio - int(ratio) > 1e-6:    # check if ratio is an int
            raise RuntimeError('resample failed, try interpolation')
        data_len = plate_data_df.shape[0]
        force_data_range = range(0, data_len, int(ratio))
        return plate_data_df.loc[force_data_range]


class HaishengSensorReader(IMUSensorReader):
    def __init__(self, file, cut_off_fre=20, filter_order=4):
        super().__init__()
        self._file = file
        self._sampling_rate = HAISHENG_SENSOR_SAMPLE_RATE
        self.data_raw_df = self._get_raw_data()
        self.data_processed_df = self._get_sensor_data_processed(cut_off_fre, filter_order)

    def _get_raw_data(self):
        data_raw_df = pd.read_csv(self._file, usecols=range(13), header=None)
        data_raw_df.columns = COLUMN_NAMES_HAISHENG
        return data_raw_df

    @staticmethod
    def rename_haisheng_sensor_files(sensor_folder, readme_xls):
        readme_sheet = xlrd.open_workbook(readme_xls).sheet_by_index(0)
        file_formal_names = readme_sheet.col_values(1)[2:]
        file_nums_foot = readme_sheet.col_values(2)[2:]
        file_nums_trunk = readme_sheet.col_values(3)[2:]
        sensor_locs = ['foot', 'trunk']
        file_nums_all = {sensor_locs[0]: file_nums_foot, sensor_locs[1]: file_nums_trunk}
        for sensor_loc in sensor_locs:
            # check if files have already been renamed
            folder_rename_str = '{path}\\{sensor_loc}_renamed'.format(path=sensor_folder, sensor_loc=sensor_loc)
            if os.path.exists(folder_rename_str):
                continue
            os.makedirs(folder_rename_str)

            # copy files
            file_nums_current_loc = file_nums_all[sensor_loc]
            for file_num, file_formal_name in zip(file_nums_current_loc, file_formal_names):
                file_ori_str = '{path}\\{sensor_loc}\\DATA_{file_num}.CSV'.format(
                    path=sensor_folder, sensor_loc=sensor_loc, file_num=int(file_num))
                file_new_str = '{rename_path}\\{file_formal_name}.csv'.format(
                    rename_path=folder_rename_str, file_formal_name=file_formal_name)
                copyfile(file_ori_str, file_new_str)




    # @staticmethod
    # def sync_with_vicon(vicon_gyr, sensor_gyr, check=False):
    #     correlation = np.correlate(vicon_gyr, sensor_gyr, 'valid')
    #     vicon_data_len = len(vicon_gyr)
    #     sensor_data_len = len(sensor_gyr)
    #     # normalize correlation data
    #     for i_sample in range(sensor_data_len - vicon_data_len + 1):
    #         norm_range = range(i_sample, i_sample+vicon_data_len)
    #         correlation[i_sample] = correlation[i_sample] / norm(sensor_gyr[norm_range])
    #
    #     if check:
    #         plt.plot(correlation)
    #         print(np.argmax(correlation) + vicon_data_len)
    #         # plt.show()
    #     diff = len(sensor_gyr) - np.argmax(correlation) - 1
    #     return diff

