"""
Notes:
    1. In leave one out testing, yangcan's result was good.
"""
from ProcessorLR import ProcessorLR
from ProcessorLRCNNv0 import ProcessorLRCNNv0
from ProcessorLRCNNv0_1 import ProcessorLRCNNv0_1
from ProcessorLRCNNv1 import ProcessorLRCNNv1
from ProcessorLRCNNv2 import ProcessorLRCNNv2
from ProcessorLRCNNv3 import ProcessorLRCNNv3
from ProcessorLRCNNv3_1 import ProcessorLRCNNv3_1
from ProcessorLRCNNv4 import ProcessorLRCNNv4
from ProcessorLRCNNv5 import ProcessorLRCNNv5
from const import SUB_NAMES, HAISHENG_SENSOR_SAMPLE_RATE, TRIAL_NAMES, SUB_AND_RUNNING_TRIALS, \
    SUB_AND_NIKE_TRIALS, SUB_AND_MINI_TRIALS, RUNNING_TRIALS, MINI_TRIALS, _24_TRIALS, _28_TRIALS

test = {'190521GongChangyang': _24_TRIALS, '190523ZengJia': _24_TRIALS, '190518MouRongzi': _24_TRIALS,
         '190522QinZhun': _24_TRIALS, '190522YangCan': _24_TRIALS, '190521LiangJie': _24_TRIALS}
train = {'190521GongChangyang': _28_TRIALS, '190523ZengJia': _28_TRIALS, '190518MouRongzi': _28_TRIALS,
        '190522QinZhun': _28_TRIALS, '190522YangCan': _28_TRIALS, '190521LiangJie': _28_TRIALS}

my_LR_processor = ProcessorLRCNNv3(train, test, 100, strike_off_from_IMU=True, split_train=False)
predict_result_all = my_LR_processor.prepare_data()
my_LR_processor.cnn_solution()

# # cross validation
# my_LR_processor = ProcessorLRCNNv5(SUB_AND_RUNNING_TRIALS, 100, strike_off_from_IMU=True, do_input_norm=True,
#                                    do_output_norm=False)
# predict_result_all = my_LR_processor.prepare_data_cross_vali()
