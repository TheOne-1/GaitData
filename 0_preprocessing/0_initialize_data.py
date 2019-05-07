from Initializer import SubjectDataInitializer
from const import RAW_DATA_PATH, PROCESSED_DATA_PATH

subject_folder = '190414WangDianxin'
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
my_initializer = SubjectDataInitializer(PROCESSED_DATA_PATH, subject_folder, readme_xls, initialize_100Hz=True,
                                        initialize_200Hz=True, initialize_1000Hz=True)

