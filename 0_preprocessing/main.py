import numpy as np
from preprocessor import ViconReader, HaishengSensorReader
from const import FOLDER_PATH
import matplotlib.pyplot as plt

file_names_haisheng = ['foot.csv', 'trunk.csv']
file_names_marker = ['static 2.csv', 'mini 24 strike index 3.csv']

file_path_haisheng = FOLDER_PATH + 'HaishengSensors\\' + file_names_haisheng[1]
my_haisheng_sensor_reader = HaishengSensorReader(file_path_haisheng)

file_path_marker = FOLDER_PATH + 'Vicon\\' + file_names_marker[0]
my_vicon_reader = ViconReader(file_path_marker)
plate_data_df = my_vicon_reader.get_plate_processed()
segment_marker_df = my_vicon_reader.get_marker_data_processed_segment('r_foot')
data_marker_segment_mat = segment_marker_df.values
