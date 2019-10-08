
from ProcessorLR import ProcessorLR
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS
import copy
from ProcessorLRTests import ProcessorIMUIndependentTower, ProcessorLR2DConv, ProcessorLRNoResampleGridSearch, \
    ProcessorLR2DMultiLayer, ProcessorLR2DAlexNet, ProcessorLR2DInception, ProcessorLROnlyNormalized \


# define the IMU sensors used for prediction
IMU_locations = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']


# define train and test subjects
train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
# train = {'190521GongChangyang': RUNNING_TRIALS, '190522YangCan':  RUNNING_TRIALS}


cross_vali_LR_processor = ProcessorLR(train, {}, IMU_locations, strike_off_from_IMU=1, do_output_norm=True)
cross_vali_LR_processor.cnn_cross_vali()

