
import numpy as np
from Drawer import Drawer, ResultReader
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


segments = ['l_shank']
result_date = '1013'

the_reader = ResultReader(result_date, segments)
trial_id_sets = [[2, 3], [5, 6], [9, 10], [12, 13]]

true_mean_values, true_std_values = [], []
pred_mean_values, pred_std_values = [], []
for trial_id_set in trial_id_sets:
    sub_true_means, sub_pred_means = [], []
    for sub_id in range(15):
        true_lr, pred_lr = the_reader.get_lr_values([sub_id], trial_id_set)
        sub_true_means.append(np.mean(true_lr))
        sub_pred_means.append(np.mean(pred_lr))
    true_mean_values.append(np.mean(sub_true_means))
    true_std_values.append(np.std(sub_true_means))
    pred_mean_values.append(np.mean(sub_pred_means))
    pred_std_values.append(np.std(sub_pred_means))

    # do t test
    _, pvalue = ttest_ind(sub_true_means, sub_pred_means)
    print('trial: ' + str(trial_id_set) + '\tpvalue: ' + str(pvalue)[:5])


Drawer.draw_compare_bars(true_mean_values, true_std_values, pred_mean_values, pred_std_values)

plt.show()


