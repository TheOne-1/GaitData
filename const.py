
COLUMN_NAMES_HAISHENG = ['hour', 'minute', 'second', 'millisecond', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y',
                         'gyr_z', 'mag_x', 'mag_y', 'mag_z', 'FPA_left', 'FPA_right']

SEGMENT_MARKERS = {'trunk': ['RAC', 'LAC', 'C7'], 'pelvis': ['RIAS', 'LIAS', 'LIPS', 'RIPS'],
                   'l_thigh': ['LTC1', 'LTC2', 'LTC3', 'LTC4', 'LFME', 'LFLE'],
                   'r_thigh': ['RTC1', 'RTC2', 'RTC3', 'RTC4', 'RFME', 'RFLE'],
                   'l_shank': ['LSC1', 'LSC2', 'LSC3', 'LSC4', 'LTAM', 'LFAL'],
                   'r_shank': ['RSC1', 'RSC2', 'RSC3', 'RSC4', 'RTAM', 'RFAL'],
                   'l_foot': ['LFM2', 'LFM5', 'LFCC'], 'r_foot': ['RFM2', 'RFM5', 'RFCC']}

FORCE_NAMES = ['force_frame', 'f_1_x', 'f_1_y', 'f_1_z', 'c_1_x', 'c_1_y', 'c_1_z',
               'f_2_x', 'f_2_y', 'f_2_z', 'c_2_x', 'c_2_y', 'c_2_z']

HAISHENG_SENSOR_SAMPLE_RATE = 100
MOCAP_SAMPLE_RATE = 240
PLATE_SAMPLE_RATE = 960

with open('configuration.txt', 'r') as config:
    FOLDER_PATH = config.readline()
