
from ProcessorLR import ProcessorLR
from const import SUB_AND_SI_SR_TRIALS, RUNNING_TRIALS, SI_SR_TRIALS
import copy
import matplotlib.pyplot as plt


# define the IMU sensors used for prediction
# 'trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot'
IMU_locations = ['l_shank', 'l_foot']


# define train and test subjects
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
train = {'190521GongChangyang': SI_SR_TRIALS, '190522YangCan':  SI_SR_TRIALS}

test_date = '1010'
test_name = test_date + '_l_shank_l_foot'
cross_vali_LR_processor = ProcessorLR(train, {}, IMU_locations, strike_off_from_IMU=1, do_output_norm=True)
cross_vali_LR_processor.cnn_cross_vali(test_date=test_date, test_name=test_name)
plt.show()

