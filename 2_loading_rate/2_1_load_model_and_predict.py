"""
To compare the result between C++ and python
"""
from ProcessorLRCNNv0 import ProcessorLRCNNv0
from ProcessorLRCNNv3_1 import ProcessorLRCNNv3_1
from const import SUB_AND_RUNNING_TRIALS, \
    RUNNING_TRIALS
import copy

# {'190521GongChangyang': RUNNING_TRIALS,
#          '190522YangCan': RUNNING_TRIALS, '190521LiangJie': RUNNING_TRIALS,
#          '190517ZhangYaqian': RUNNING_TRIALS, '190518MouRongzi': RUNNING_TRIALS, '190518FuZhinan': RUNNING_TRIALS,}

test = {'190521GongChangyang':  RUNNING_TRIALS}

#
my_LR_processor = ProcessorLRCNNv3_1(train, test, 100, strike_off_from_IMU=2, split_train=False, do_output_norm=False)
predict_result_all = my_LR_processor.prepare_data()
my_LR_processor.cnn_solution()
my_LR_processor.save_model_and_param()

# cross validation
# my_LR_processor = ProcessorLRCNNv5(SUB_AND_RUNNING_TRIALS, 100, strike_off_from_IMU=True, do_input_norm=True,
#                                    do_output_norm=False)
# predict_result_all = my_LR_processor.prepare_data_cross_vali()






