import numpy as np
from Drawer import Drawer, ResultReader
import matplotlib.pyplot as plt

segments = ['l_shank']
result_date = '1013'
subject_id = 0      # 0, 13

the_reader = ResultReader(result_date, segments)

# trial_id_sets = [[2, 3], [5, 6], [9, 10], [12, 13]]
trial_ids = [2, 3, 5, 6, 9, 10, 12, 13]

true_lr_list, pred_lr_list = [], []
for trial_id in trial_ids:
    true_lr, pred_lr = the_reader.get_lr_values([subject_id], [trial_id])
    true_lr_list.append(true_lr)
    pred_lr_list.append(pred_lr)

pearson_cor, _ = the_reader.get_param_mean_std_of_trial_mean('pearson correlation', ['All trials'], sub_id_list=[subject_id])
NRMSE, _ = the_reader.get_one_trial_NRMSE_mean_std(sub_id_list=[subject_id])
MAE, _ = the_reader.get_param_mean_std_of_trial_mean('absolute mean error', ['All trials'], sub_id_list=[subject_id])

Drawer.draw_example_result(true_lr_list, pred_lr_list, title='')
print(str(pearson_cor)[:4] + '\t' + str(NRMSE)[:4] + '\t' + str(MAE)[:4])
plt.show()
