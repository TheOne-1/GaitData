
from ProcessorLR import ProcessorLR
from ProcessorLRCrossVali import ProcessorLRCrossVali
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS
import copy


IMU_locations = ['trunk', 'l_foot']


train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
del train['190522QinZhun']
# train = {'190521GongChangyang': RUNNING_TRIALS[:1]}

test = {'190522QinZhun':  RUNNING_TRIALS[:1]}

my_LR_processor = ProcessorLR(train, test, IMU_locations, strike_off_from_IMU=2, do_output_norm=True)
predict_result_all = my_LR_processor.prepare_data()
my_LR_processor.cnn_solution()


# train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
# del train['190513YangYicheng']
# del train['190513OuYangjue']
# # train = {'190521GongChangyang': RUNNING_TRIALS,
# #          '190522YangCan': RUNNING_TRIALS}
# # cross validation
# my_LR_processor = ProcessorLRCNNv5(train, 200, strike_off_from_IMU=2, do_input_norm=True, do_output_norm=True)
# my_LR_processor.prepare_data_cross_vali()
