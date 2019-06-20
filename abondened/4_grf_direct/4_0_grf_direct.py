"""
Directly use each IMU data sample to estimate GRF
"""
from ProcessorDirect import ProcessorDirect
from const import SUB_NAMES, HAISHENG_SENSOR_SAMPLE_RATE, TRIAL_NAMES, SUB_AND_RUNNING_TRIALS, \
    SUB_AND_NIKE_TRIALS, SUB_AND_MINI_TRIALS, RUNNING_TRIALS, MINI_TRIALS, _24_TRIALS, _28_TRIALS

train = {'190521GongChangyang': RUNNING_TRIALS,
         '190522QinZhun': RUNNING_TRIALS, '190522YangCan': RUNNING_TRIALS, '190521LiangJie': RUNNING_TRIALS,
         '190517ZhangYaqian': RUNNING_TRIALS, '190518MouRongzi': RUNNING_TRIALS, '190518FuZhinan': RUNNING_TRIALS,}
test = {'190523ZengJia': RUNNING_TRIALS}

my_LR_processor = ProcessorDirect(train, test, 100, strike_off_from_IMU=True, split_train=False)
my_LR_processor.prepare_data()
my_LR_processor.GBDT_solution()

# # cross validation
# my_LR_processor = ProcessorLRCNNv5(train, 100, strike_off_from_IMU=True, do_input_norm=True,
#                                    do_output_norm=False)
# predict_result_all = my_LR_processor.prepare_data_cross_vali()
