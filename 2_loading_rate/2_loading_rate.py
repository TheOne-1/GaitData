from ProcessorLR import ProcessorLR
from const import SUB_NAMES, HAISHENG_SENSOR_SAMPLE_RATE, TRIAL_NAMES

subject_name = SUB_NAMES[1]
trials = TRIAL_NAMES[1:7]
# trials = TRIAL_NAMES[1:7] + TRIAL_NAMES[8:14]
sub_and_trials = {'190414WangDianxin': trials, '190423LiuSensen': trials, '190424XuSen': trials}
# sub_and_trials = {'190414WangDianxin': trials}
my_LR_processor = ProcessorLR(sub_and_trials, sub_and_trials, HAISHENG_SENSOR_SAMPLE_RATE, strike_off_from_IMU=True)
