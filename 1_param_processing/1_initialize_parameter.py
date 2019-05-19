from const import PROCESSED_DATA_PATH, RAW_DATA_PATH, SUB_NAMES
from ParameterProcessor import ParamProcessor

subject_folder = SUB_NAMES[2]
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
my_initializer = ParamProcessor(subject_folder, readme_xls, check_steps=False, plot_strike_off=False)
my_initializer.start_initalization(PROCESSED_DATA_PATH)
