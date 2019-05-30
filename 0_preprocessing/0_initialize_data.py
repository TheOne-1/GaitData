from Initializer import SubjectDataInitializer
from const import RAW_DATA_PATH, PROCESSED_DATA_PATH, SUB_NAMES, SUB_AND_TRIALS, TRIAL_NAMES

subject_folder = SUB_NAMES[8]
trials_100hz = SUB_AND_TRIALS[subject_folder]
readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
my_initializer = SubjectDataInitializer(PROCESSED_DATA_PATH, subject_folder, trials_100hz, readme_xls,
                                        initialize_100Hz=True,
                                        initialize_200Hz=True, initialize_1000Hz=True, check_sync=True,
                                        check_running_period=True)
