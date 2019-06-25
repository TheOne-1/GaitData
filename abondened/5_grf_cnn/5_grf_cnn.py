from const import SUB_AND_RUNNING_TRIALS, RUNNING_TRIALS
import copy
from ProcessorGRF import ProcessorGRF
from ProcessorGRFv1 import ProcessorGRFv1


# train = {'190521GongChangyang': RUNNING_TRIALS,
#          '190522YangCan': RUNNING_TRIALS, '190521LiangJie': RUNNING_TRIALS,
#          '190517ZhangYaqian': RUNNING_TRIALS, '190518MouRongzi': RUNNING_TRIALS, '190518FuZhinan': RUNNING_TRIALS,}
train = {'190521GongChangyang': RUNNING_TRIALS}
# train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
# del train['190522QinZhun']
# del train['190513OuYangjue']
# del train['190517ZhangYaqian']
test = {'190522QinZhun':  RUNNING_TRIALS}

my_GRF_processor = ProcessorGRFv1(train, test, 100, strike_off_from_IMU=2)
my_GRF_processor.prepare_data()
my_GRF_processor.export_model()
