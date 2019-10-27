from ProcessorLR import ProcessorLR
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS, SUB_AND_SI_SR_TRIALS, \
    SI_SR_TRIALS
import copy
from ProcessorLRTests import ProcessorIMUIndependentTower, ProcessorLR2DConv, ProcessorLRNoResampleGridSearch, \
    ProcessorLR2DMultiLayer, ProcessorLR2DAlexNet, ProcessorLR2DInception, ProcessorLROnlyNormalized, \
    ProcessorLRCrazyKernel, ProcessorLRNoResample, ProcessorLR2DConvGridSearch

# define the IMU sensors used for prediction
# 'trunk', 'pelvis', 'l_thigh', 'l_shank',
IMU_locations = ['l_shank']

# define train and test subjects  190423LiuSensen
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
del train['190517ZhangYaqian']
test = {'190517ZhangYaqian': SI_SR_TRIALS}

# train = {'190517ZhangYaqian':  SI_SR_TRIALS[:1]}
# test = {'190517ZhangYaqian':  SI_SR_TRIALS[:1]}

base_LR_processor = ProcessorLR(train, test, IMU_locations, strike_off_from_IMU=1,
                                          do_input_norm=True, do_output_norm=True)
base_LR_processor.cnn_train_test()
