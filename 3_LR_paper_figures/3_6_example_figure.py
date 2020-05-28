import numpy as np
from Drawer import Drawer, ResultReader
import matplotlib.pyplot as plt
from const import SUB_ID_PAPER, SI_SR_TRIALS, TRIAL_NAMES
import random


segments = ['l_shank']
result_date = '1028'

trial_ids = [int(TRIAL_NAMES.index(trial_name)) for trial_name in SI_SR_TRIALS]
the_reader = ResultReader(result_date, segments)
for subject_id in SUB_ID_PAPER[:1]:

    true_lr_list, pred_lr_list = [], []
    for trial_id in trial_ids:
        true_lr, pred_lr = the_reader.get_lr_values([subject_id], [trial_id])
        true_lr, pred_lr = true_lr.values, pred_lr.values

        # 0 for use the whole trial; 1 for take 10 steps from the start, middle, and end;
        # 2 for randomly select 25 steps from one trial, 200 steps in total
        steps_to_show = 2
        if steps_to_show == 0:
            # use the whole trial
            true_lr_list.append(true_lr)
            pred_lr_list.append(pred_lr)
        elif steps_to_show == 1:
            # only take a the first, middle and last 10 steps from the trial
            middle_step = int(len(true_lr)/2)
            true_lr_list.append(np.concatenate([true_lr[:10], true_lr[middle_step-5:middle_step+5], true_lr[-10:]]))
            pred_lr_list.append(np.concatenate([pred_lr[:10], pred_lr[middle_step-5:middle_step+5], pred_lr[-10:]]))
        elif steps_to_show == 2:
            true_lr_current, pred_lr_current = zip(*random.sample(list(zip(true_lr, pred_lr)), 20))
            true_lr_list.append(true_lr_current)
            pred_lr_list.append(pred_lr_current)

    pearson_cor, _ = the_reader.get_param_mean_std_of_trial_mean('pearson correlation', ['All trials'], sub_id_list=[subject_id])
    NRMSE, _ = the_reader.get_one_trial_NRMSE_mean_std(sub_id_list=[subject_id])
    MAE, _ = the_reader.get_param_mean_std_of_trial_mean('absolute mean error', ['All trials'], sub_id_list=[subject_id])

    Drawer.draw_example_result(true_lr_list, pred_lr_list, title='')
    print(str(pearson_cor)[:4] + '\t' + str(NRMSE)[:4] + '\t' + str(MAE)[:4])

plt.show()
