from ProcessorLR import ProcessorLR
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS, SUB_AND_RUNNING_TRIALS_NO_BASELINE, \
    RUNNING_TRIALS_NO_BASELINE
import copy
from ProcessorLRTests import ProcessorIMUIndependentTower, ProcessorLR2DConv, ProcessorLRNoResampleGridSearch, \
    ProcessorLR2DMultiLayer, ProcessorLR2DAlexNet, ProcessorLR2DInception, ProcessorLROnlyNormalized, \
    ProcessorLRCrazyKernel, ProcessorLRNoResample

# define the IMU sensors used for prediction
# 'trunk', 'pelvis', 'l_thigh', 'l_shank',
IMU_locations = ['l_shank', 'l_foot']

# define train and test subjects
train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
del train['190522QinZhun']
del train['190522YangCan']
del train['190522SunDongxiao']
test = {'190522QinZhun': RUNNING_TRIALS, '190522YangCan': RUNNING_TRIALS,
        '190522SunDongxiao': RUNNING_TRIALS}

# train = {'190522QinZhun':  RUNNING_TRIALS[:1]}
# test = {'190522SunDongxiao':  RUNNING_TRIALS[:1]}

base_LR_processor = ProcessorLRNoResample(train, test, IMU_locations, strike_off_from_IMU=1,
                                                    do_input_norm=True, do_output_norm=True)
base_LR_processor.cnn_train_test()
