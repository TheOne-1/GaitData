from ProcessorLR import ProcessorLR
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS, SUB_AND_SI_SR_TRIALS, \
    SI_SR_TRIALS
import copy
from ProcessorLRTests import ProcessorIMUIndependentTower, ProcessorLR2DConv, ProcessorLRNoResampleGridSearch, \
    ProcessorLR2DMultiLayer, ProcessorLR2DAlexNet, ProcessorLR2DInception, ProcessorLROnlyNormalized, \
    ProcessorLRCrazyKernel, ProcessorLRNoResample, ProcessorLR2DConvGridSearch

# define the IMU sensors used for prediction
# 'trunk', 'pelvis', 'l_thigh', 'l_shank',
IMU_locations = ['l_shank', 'l_foot']

# define train and test subjects
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
del train['190522QinZhun']
del train['190522YangCan']
del train['190522SunDongxiao']
test = {'190522QinZhun': SI_SR_TRIALS, '190522YangCan': SI_SR_TRIALS,
        '190522SunDongxiao': SI_SR_TRIALS}

train = {'190414WangDianxin':  SI_SR_TRIALS[7:8]}
test = {'190522QinZhun':  SI_SR_TRIALS[:1]}

base_LR_processor = ProcessorLR(train, test, IMU_locations, strike_off_from_IMU=1,
                                                do_input_norm=True, do_output_norm=True)
base_LR_processor.cnn_train_test()
