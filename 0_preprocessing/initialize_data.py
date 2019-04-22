import numpy as np
from Initializer import SubjectDataInitializer
from ViconReader import ViconReader
from HaishengSensorReader import HaishengSensorReader
from XsensReader import XsensReader
from GyrSimulator import GyrSimulator
from const import RAW_DATA_PATH, FILE_NAMES, XSENS_FILE_NAME_DIC, HAISHENG_SENSOR_SAMPLE_RATE, MOCAP_SAMPLE_RATE, \
    PROCESSED_DATA_PATH

subject_folder = '190414WangDianxin'
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
my_initializer = SubjectDataInitializer(PROCESSED_DATA_PATH, subject_folder, readme_xls)
x=1

