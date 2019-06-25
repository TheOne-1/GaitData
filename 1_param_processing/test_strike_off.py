from const import PROCESSED_DATA_PATH, RAW_DATA_PATH
from ParameterProcessor import ParamProcessor

subject_folder = '190414WangDianxin'
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'


my_initializer = ParamProcessor(subject_folder, readme_xls, check_steps=False, plot_strike_off=False)
my_initializer.test_strike_off(PROCESSED_DATA_PATH)

