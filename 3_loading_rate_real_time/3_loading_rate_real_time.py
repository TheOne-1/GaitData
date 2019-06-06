from ProcessorLRRealTime import ProcessorLRRealTime
from ProcessorLRRealTimeCNN import ProcessorLRRealTimeCNN
from const import SUB_NAMES, HAISHENG_SENSOR_SAMPLE_RATE, TRIAL_NAMES, SUB_AND_RUNNING_TRIALS, \
    SUB_AND_NIKE_TRIALS, SUB_AND_MINI_TRIALS, RUNNING_TRIALS, MINI_TRIALS, _24_TRIALS, _28_TRIALS


train = {'190521GongChangyang': RUNNING_TRIALS,
         '190522QinZhun': RUNNING_TRIALS, '190522YangCan': RUNNING_TRIALS, '190521LiangJie': RUNNING_TRIALS,
         '190517ZhangYaqian': RUNNING_TRIALS, '190518MouRongzi': RUNNING_TRIALS, '190518FuZhinan': RUNNING_TRIALS,}
test = {'190523ZengJia': RUNNING_TRIALS}

my_LR_processor = ProcessorLRRealTimeCNN(train, test, 100, split_train=False)
my_LR_processor.prepare_data()
my_LR_processor.cnn_solution()



































