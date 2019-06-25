from Initializer import ViconReader, HaishengSensorReader, GyrSimulator
from const import RAW_DATA_PATH, TRIAL_NAMES, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE, \
    PROCESSED_DATA_PATH
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os


subject_folder = '190414WangDianxin'
# create folder for this subject
if not os.path.exists(PROCESSED_DATA_PATH + subject_folder):
    os.makedirs(PROCESSED_DATA_PATH + subject_folder)

for trial_name in TRIAL_NAMES[1:]:
    file_path_xsens = '{path}{sub_folder}\\{sensor}\\{trial_folder}\\'.format(
        path=RAW_DATA_PATH, sub_folder=subject_folder, sensor='xsens', trial_folder=trial_name)
    file_path_vicon = '{path}{sub_folder}\\{sensor}\\{file_name}.csv'.format(
        path=RAW_DATA_PATH, sub_folder=subject_folder, sensor='vicon', file_name=trial_name)
    file_path_haisheng = '{path}{sub_folder}\\{sensor}\\{sensor_loc}\\{trial_name}.csv'.format(
        path=RAW_DATA_PATH, sub_folder=subject_folder, sensor='haisheng', sensor_loc='r_foot_renamed', trial_name=trial_name)
    readme_xls_path = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'

    HaishengSensorReader.rename_haisheng_sensor_files(RAW_DATA_PATH + subject_folder + '\\haisheng', readme_xls_path)
    my_haisheng_sensor_reader = HaishengSensorReader(file_path_haisheng)
    gyr_norm_haisheng = my_haisheng_sensor_reader.get_normalized_gyr('gyr_z')

    my_vicon_reader = ViconReader(file_path_vicon)
    marker_df_r_foot = my_vicon_reader.get_marker_data_processed_segment('r_foot')
    marker_df_r_foot = ViconReader.resample_data(marker_df_r_foot, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE)

    my_nike_gyr_simulator = GyrSimulator(subject_folder, 'r_foot')
    gyr_vicon = my_nike_gyr_simulator.get_gyr(trial_name, marker_df_r_foot, sampling_rate=HAISHENG_SENSOR_SAMPLE_RATE)
    gyr_norm_vicon = norm(gyr_vicon, axis=1)
    vicon_delay = GyrSimulator._sync_running_foot(gyr_norm_vicon, gyr_norm_haisheng)
    # print(vicon_delay)
    if vicon_delay > 0:
        plt.figure()
        plt.plot(gyr_norm_haisheng[vicon_delay:])
        plt.plot(gyr_norm_vicon)
    else:
        plt.figure()
        plt.plot(gyr_norm_haisheng)
        plt.plot(gyr_norm_vicon[-vicon_delay:])

    plt.show()

