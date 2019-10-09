
from ProcessorLR import ProcessorLR
from const import SUB_AND_SI_SR_TRIALS, RUNNING_TRIALS
import copy
from ProcessorLRTests import ProcessorIMUIndependentTower, ProcessorLR2DConv, ProcessorLRNoResample, \
    ProcessorLR2DMultiLayer, ProcessorLR2DInception \


# define the IMU sensors used for prediction
# 'trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot'
IMU_locations = ['l_shank', 'l_foot']


# define train and test subjects
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
# train = {'190521GongChangyang': RUNNING_TRIALS[:1], '190522YangCan':  RUNNING_TRIALS[:1]}


cross_vali_LR_processor = ProcessorLR(train, {}, IMU_locations, strike_off_from_IMU=1, do_output_norm=True)
cross_vali_LR_processor.cnn_cross_vali()

