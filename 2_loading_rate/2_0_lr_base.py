
from ProcessorLR import ProcessorLR
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS, TRIAL_NAMES
import copy


# define the IMU sensors used for prediction
IMU_locations = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']


# define train and test subjects
train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
del train['190522QinZhun']
del train['190522YangCan']
del train['190522SunDongxiao']
test = {'190522QinZhun':  RUNNING_TRIALS, '190522YangCan': RUNNING_TRIALS, '190522SunDongxiao': RUNNING_TRIALS}

# train = {'190522QinZhun': RUNNING_TRIALS}
# test = {'190522YangCan':  RUNNING_TRIALS}
base_LR_processor = ProcessorLR(train, test, IMU_locations, strike_off_from_IMU=2, do_output_norm=True)
base_LR_processor.cnn_train_test()

