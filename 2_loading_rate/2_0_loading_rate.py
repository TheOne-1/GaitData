"""
Notes:
    1. In leave one out testing, yangcan's result was good.
"""
from ProcessorLRCNNv0 import ProcessorLRCNNv0
from ProcessorLRCNNv3_1 import ProcessorLRCNNv3_1
from ProcessorLRCNNv3_2 import ProcessorLRCNNv3_2
from ProcessorLRCNNv5 import ProcessorLRCNNv5
from const import SUB_AND_RUNNING_TRIALS, \
    RUNNING_TRIALS
import copy

# # train = {'190521GongChangyang': RUNNING_TRIALS}
# train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
# del train['190522QinZhun']
# del train['190517FuZhenzhen']
# del train['190513OuYangjue']
# test = {'190522QinZhun':  RUNNING_TRIALS}
#
# my_LR_processor = ProcessorLRCNNv3_1(train, test, 100, strike_off_from_IMU=2, split_train=False, do_output_norm=False)
# predict_result_all = my_LR_processor.prepare_data()
# my_LR_processor.cnn_solution()
# my_LR_processor.save_model()



train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
del train['190517FuZhenzhen']
del train['190513OuYangjue']
# train = {'190521GongChangyang': RUNNING_TRIALS,
#          '190522YangCan': RUNNING_TRIALS}
# cross validation
my_LR_processor = ProcessorLRCNNv5(train, 100, strike_off_from_IMU=2, do_input_norm=True)
predict_result_all = my_LR_processor.prepare_data_cross_vali()
