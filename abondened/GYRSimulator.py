import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate as interpo
from scipy.signal import butter, filtfilt
from transforms3d.euler import mat2euler
from scipy.spatial.distance import cosine
from numpy.linalg import norm
from const import MOCAP_SAMPLE_RATE


class GYRSimulator:
    @staticmethod
    def get_orientation_diff(vector_g_sensor, vector_rotat_sensor_all, vector_g_simu, vector_rota_simu_all):
        data_len = min(vector_rotat_sensor_all.shape[0], vector_rota_simu_all.shape[0])
        euler_diff = []

        # plt.plot(vector_rotat_sensor_all[:, 0])
        # plt.plot(vector_rota_simu_all[:, 0])
        # plt.show()

        for i_sample in range(data_len):
            if norm(vector_rota_simu_all[i_sample, :]) < 1 or norm(vector_rota_simu_all[i_sample, :]) > 3:
                continue

            vector_rota_sensor = vector_rotat_sensor_all[i_sample, :]
            vector_rota_simu = vector_rota_simu_all[i_sample, :]

            vector_extra_sensor = np.cross(vector_g_sensor, vector_rota_sensor)
            vector_extra_simu = np.cross(vector_g_simu, vector_rota_simu)

            dcm = np.array([[1 - cosine(vector_extra_sensor, vector_extra_simu), 1 - cosine(vector_extra_sensor, vector_rota_simu), 1 - cosine(vector_extra_sensor, vector_g_simu)],
                            [1 - cosine(vector_rota_sensor, vector_extra_simu), 1 - cosine(vector_rota_sensor, vector_rota_simu), 1 - cosine(vector_rota_sensor, vector_g_simu)],
                            [1 - cosine(vector_g_sensor, vector_extra_simu), 1 - cosine(vector_g_sensor, vector_rota_simu), 1 - cosine(vector_g_sensor, vector_g_simu)]])
            euler_diff.append(np.rad2deg(mat2euler(dcm)))
        euler_diff = np.array(euler_diff)
        return euler_diff

    @staticmethod
    def compare_gyr(gyr_simu, gyr_xsens, xsens_delay=0):
        plt.figure()
        plt.plot(gyr_simu)
        plt.plot(gyr_xsens[xsens_delay:])

    # return how much array_1 is ahead of array_0
    @staticmethod
    def sync_data(array_0, array_1, check=False):
        correlation = np.correlate(array_0, array_1, 'full')
        if check:
            plt.plot(correlation)
            print(np.argmax(correlation) - len(array_0))
            # plt.show()
        diff = len(array_1) - np.argmax(correlation) - 1
        return diff

    @staticmethod
    def get_marker_cali_matrix(vicon_data, static_start, static_end):
        """
        standing marker data for IMU simulation calibration
        :param vicon_data:
        :param static_start: unit: second
        :param static_end: unit: second
        :return:
        """
        static_start_sample = MOCAP_SAMPLE_RATE * static_start
        static_end_sample = MOCAP_SAMPLE_RATE * static_end
        cali_period_data = vicon_data[static_start_sample:static_end_sample, :]
        vicon_data_average = np.mean(cali_period_data, axis=0)
        cali_matrix = vicon_data_average.reshape([-1, 3])
        return cali_matrix

    # get virtual marker and R_IMU_transform
    @staticmethod
    def get_virtual_marker(simulated_marker, vicon_data, marker_cali_matrix, R_standing_to_ground):
        segment_marker_num = marker_cali_matrix.shape[0]
        data_len = vicon_data.shape[0]
        virtual_marker = np.zeros([data_len, 3])
        R_IMU_transform = np.zeros([3, 3, data_len])
        for i_frame in range(data_len):
            current_marker_matrix = vicon_data[i_frame, :].reshape([segment_marker_num, 3])
            [R_between_frames, t] = GYRSimulator.rigid_transform_3D(marker_cali_matrix, current_marker_matrix)
            virtual_marker[i_frame, :] = (np.dot(R_between_frames, simulated_marker) + t)
            R_IMU_transform[:, :, i_frame] = np.matmul(R_standing_to_ground, R_between_frames.T)
        return virtual_marker, R_IMU_transform

    @staticmethod
    def rigid_transform_3D(A, B):
        assert len(A) == len(B)

        N = A.shape[0]  # total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        # centre the points
        AA = A - np.tile(centroid_A, (N, 1))
        BB = B - np.tile(centroid_B, (N, 1))
        # dot is matrix multiplication for array
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            # print
            # "Reflection detected"
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = -np.dot(R, centroid_A.T) + centroid_B.T
        return R, t

    @staticmethod
    def data_filt(data, cut_off_fre, filter_order=4):
        fre = cut_off_fre / (MOCAP_SAMPLE_RATE/2)
        b, a = butter(filter_order, fre, 'lowpass')
        data_filt = filtfilt(b, a, data, axis=0)
        return data_filt

    @staticmethod
    def get_gyr(walking_data_df, R_IMU_transform):
        walking_data = walking_data_df
        data_len = walking_data.shape[0]
        marker_number = int(walking_data.shape[1] / 3)
        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        gyr_middle = np.zeros([data_len, 3])
        for i_frame in range(data_len - 1):
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame + 1, :].reshape([marker_number, 3])
            [R_one_sample, t] = GYRSimulator.rigid_transform_3D(current_marker_matrix, next_marker_matrix)
            theta = np.math.acos((np.matrix.trace(R_one_sample) - 1) / 2)
            a, b = np.linalg.eig(R_one_sample)
            for i_eig in range(a.__len__()):
                if abs(a[i_eig].imag) < 1e-12:
                    vector = b[:, i_eig].real
                    break
                if i_eig == a.__len__():
                    raise RuntimeError('no eig')

            if (R_one_sample[2, 1] - R_one_sample[1, 2]) * vector[0] < 0:  # check the direction of the rotation axis
                vector = -vector
            vector = np.dot(R_IMU_transform[:, :, i_frame].T, vector)
            gyr_middle[i_frame, :] = theta * vector * MOCAP_SAMPLE_RATE

        step_middle = np.arange(0.5 / MOCAP_SAMPLE_RATE, data_len / MOCAP_SAMPLE_RATE + 0.5 / MOCAP_SAMPLE_RATE,
                                1 / MOCAP_SAMPLE_RATE)
        step_gyr = np.arange(0, data_len / MOCAP_SAMPLE_RATE, 1 / MOCAP_SAMPLE_RATE)
        # in splprep, s the amount of smoothness. 6700 might be appropriate
        tck, step = interpo.splprep(gyr_middle.T, u=step_middle, s=0)
        gyr = interpo.splev(step_gyr, tck, der=0)
        gyr = np.column_stack([gyr[0], gyr[1], gyr[2]])

        # plt.figure()
        # plt.subplot(221)
        # plt.plot(gyr[:, 0])
        # plt.subplot(222)
        # plt.plot(gyr[:, 1])
        # plt.subplot(223)
        # plt.plot(gyr[:, 2])
        # gyr_norm = np.linalg.norm(gyr, axis=1)
        # plt.subplot(224)
        # plt.plot(gyr_norm)
        # plt.show()

        return gyr

    @staticmethod
    def get_segment_R():
        return np.eye(3)


