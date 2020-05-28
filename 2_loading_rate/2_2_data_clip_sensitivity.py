from ProcessorLR import ProcessorLR
from const import SUB_AND_SI_SR_TRIALS, RUNNING_TRIALS, SI_SR_TRIALS
import copy
import matplotlib.pyplot as plt

date = '1029'
segment = 'l_shank'
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)

cross_vali_LR_processor = ProcessorLR(train, {}, [segment], strike_off_from_IMU=2)
cross_vali_LR_processor.cnn_cross_vali(test_date=date, test_name=date + '_' + segment, plot=False)




