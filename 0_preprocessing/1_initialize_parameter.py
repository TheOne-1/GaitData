from const import PROCESSED_DATA_PATH, RAW_DATA_PATH, FILE_NAMES
from ParameterProcessor import ParamProcessor

subject_folder = '190423LiuSensen'
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
my_initializer = ParamProcessor(PROCESSED_DATA_PATH, subject_folder, readme_xls, check_steps=True, plot_strike_off=True)
