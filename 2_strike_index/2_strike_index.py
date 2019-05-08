from ProcessorSI import ProcessorSI
from const import HAISHENG_SENSOR_SAMPLE_RATE

subject_name = '190414WangDianxin'
my_LR_processor = ProcessorSI(subject_name, 'r', HAISHENG_SENSOR_SAMPLE_RATE)
my_LR_processor.linear_regression_solution()
