
from ProcessorLRLoadModel import ProcessorLRLoadModel
from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS
import copy

# 190522QinZhun, 190517ZhangYaqian, 190523ZengJia
test = {'190517ZhangYaqian':  RUNNING_TRIALS[0:2]}

my_LR_processor = ProcessorLRLoadModel(test, 200, strike_off_from_IMU=2, split_train=False, do_output_norm=True)
predict_result_all = my_LR_processor.prepare_data()
my_LR_processor.cnn_solution()
