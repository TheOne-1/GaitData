from ProcessorLR import ProcessorLR
from const import SUB_AND_SI_SR_TRIALS, SI_SR_TRIALS
import copy
from ComboGenerator import ComboGenerator


date = '1029'
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
train = {'190521GongChangyang': SI_SR_TRIALS, '190522QinZhun':  SI_SR_TRIALS}

ComboGenerator.create_folders(date)
segment = 'l_shank'
cross_vali_LR_processor = ProcessorLR(train, {}, [segment], strike_off_from_IMU=2)
cross_vali_LR_processor.cnn_cross_vali(test_date=date, test_name=date + '_' + segment, plot=False)
