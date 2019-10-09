from Initializer import SubjectDataInitializer
from const import RAW_DATA_PATH, PROCESSED_DATA_PATH, SUB_NAMES, SUB_AND_TRIALS, TRIAL_NAMES

for subject_folder in SUB_NAMES:
    trials_to_init = SUB_AND_TRIALS[subject_folder][7:8]
    readme_xls = RAW_DATA_PATH + subject_folder + '\\readme\\readme.xlsx'
    my_initializer = SubjectDataInitializer(PROCESSED_DATA_PATH, subject_folder, trials_to_init, readme_xls,
                                            initialize_100Hz=False,
                                            initialize_200Hz=True, initialize_1000Hz=True, check_sync=False,
                                            check_running_period=False)
