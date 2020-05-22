import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from const import DATA_COLUMNS_XSENS, MOCAP_SAMPLE_RATE, DATA_COLUMNS_IMU, XSENS_ROTATION_CORRECTION_NIKE
import xsensdeviceapi.xsensdeviceapi_py36_64 as xda
from threading import Lock
import time
from IMUSensorReader import IMUSensorReader


class XsensReader(IMUSensorReader):
    def __init__(self, file, subject_folder, sensor_loc, trial_name):
        super().__init__()
        self._file = file
        self._subject_folder = subject_folder
        self._sensor_loc = sensor_loc
        self._trial_name = trial_name
        self._sampling_rate = MOCAP_SAMPLE_RATE
        device, callback, control = self.__initialize_device()
        self.data_raw_df = self._get_raw_data(device, callback, control)
        self.data_processed_df = self._get_sensor_data_processed()

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
        # rotation correction
        do_rotation = False
        if 'nike' in self._trial_name:
            if self._subject_folder in XSENS_ROTATION_CORRECTION_NIKE.keys():
                sub_rota_dict = XSENS_ROTATION_CORRECTION_NIKE[self._subject_folder]
                if self._sensor_loc in sub_rota_dict.keys():
                    do_rotation = True
                    rotation_mat = sub_rota_dict[self._sensor_loc]

        # Export the data
        data_mat = np.zeros([packetCount, 13])
        for index in range(packetCount):
            # Retrieve a packet
            packet = device.getDataPacketByIndex(index)
            acc = packet.calibratedAcceleration()
            gyr = packet.calibratedGyroscopeData()
            mag = packet.calibratedMagneticField()

            # skip the mag if it does not exist
            if len(mag) != 3:
                data_mat[index, 0:6] = np.concatenate([acc, gyr])
                data_mat[index, 6:9] = data_mat[index - 1, 6:9]
            else:
                data_mat[index, 0:9] = np.concatenate([acc, gyr, mag])
            # correct the placement error
            if do_rotation:
                for i_channel in range(3):
                    data_mat[index, 3*i_channel:3*(i_channel+1)] = np.matmul(
                        rotation_mat, data_mat[index, 3*i_channel:3*(i_channel+1)])

            quaternion = packet.orientationQuaternion()
            data_mat[index, 9:13] = quaternion
        data_raw_df = pd.DataFrame(data_mat)
        data_raw_df.columns = DATA_COLUMNS_XSENS
        device.removeCallbackHandler(callback)
        control.close()
        return data_raw_df

    def _get_sensor_data_processed(self):
        data_raw = self.data_raw_df[DATA_COLUMNS_IMU]
        data_processed_df = pd.DataFrame(data_raw, columns=DATA_COLUMNS_IMU)
        return data_processed_df


class XsensReaderLjx(XsensReader):
    def _get_raw_data(self, device, callback, control):
        # Get total number of samples
        packetCount = device.getDataPacketCount()
        # Export the data
        data_mat = np.zeros([packetCount, 16])
        for index in range(packetCount):
            # Retrieve a packet
            packet = device.getDataPacketByIndex(index)
            acc = packet.calibratedAcceleration()
            gyr = packet.calibratedGyroscopeData()
            mag = packet.calibratedMagneticField()

            # skip the mag if it does not exist
            if len(mag) != 3:
                data_mat[index, 0:6] = np.concatenate([acc, gyr])
                data_mat[index, 6:9] = data_mat[index - 1, 6:9]
            else:
                data_mat[index, 0:9] = np.concatenate([acc, gyr, mag])

            angles = packet.orientationEuler()
            data_mat[index, 9:12] = [angles.x(), angles.y(), angles.z()]
            quat = packet.orientationQuaternion()
            data_mat[index, 12:16] = quat

        data_raw_df = pd.DataFrame(data_mat)
        data_raw_df.columns = ['l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
                               'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
                               'l_thigh_mag_x', 'l_thigh_mag_y', 'l_thigh_mag_z',
                               'l_thigh_roll', 'l_thigh_pitch', 'l_thigh_yaw',
                               'l_thigh_q0', 'l_thigh_q1', 'l_thigh_q2', 'l_thigh_q3']
        device.removeCallbackHandler(callback)
        control.close()
        return data_raw_df

    def _get_sensor_data_processed(self):
        data_processed_df = self.data_raw_df
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
