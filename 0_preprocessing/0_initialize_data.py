from Initializer import SubjectDataInitializer
from const import RAW_DATA_PATH, PROCESSED_DATA_PATH, SUB_NAMES

# 190414WangDianxin, 190423LiuSensen, 190424XuSen, 190426YuHongzhe
subject_folder = SUB_NAMES[3]
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
my_initializer = SubjectDataInitializer(PROCESSED_DATA_PATH, subject_folder, readme_xls, initialize_100Hz=True,
                                        initialize_200Hz=True, initialize_1000Hz=True, check_sync=False,
                                        check_running_period=False)
