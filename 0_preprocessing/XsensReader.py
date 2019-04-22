import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from const import DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE, DATA_COLUMNS_IMU
import xsensdeviceapi.xsensdeviceapi_py36_64 as xda
from threading import Lock
import time
from IMUSensorReader import IMUSensorReader


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
        assert (control is not 0)  # check if control is 0; raise Exception if not.
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
        assert (device is not 0)  # check if device is found; raise Exception if not.

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
        device.loadLogFile()
        while callback.progress() != 100:
            time.sleep(0.01)
        return device, callback, control

    def _get_raw_data(self, device, callback, control):
        # Get total number of samples
        packetCount = device.getDataPacketCount()
        # Export the data
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
        b_IMU, a_IMU = butter(filter_order, wn, btype='low')
        data_processed = data_raw.values  # Frame column does not need a filter
        data_processed = filtfilt(b_IMU, a_IMU, data_processed, axis=0)  # filtering
        data_processed_df = pd.DataFrame(data_processed, columns=DATA_COLUMNS_IMU)
        return data_processed_df


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
