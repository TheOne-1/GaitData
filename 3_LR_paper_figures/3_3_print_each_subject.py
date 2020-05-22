from Drawer import Drawer, ResultReader
from const import SUB_NAMES, RUNNING_TRIALS, _24_TRIALS, _28_TRIALS, NIKE_TRIALS, MINI_TRIALS


result_date = '1028'
reader_cnn = ResultReader(result_date, ['l_shank'])
reader_pta = ResultReader(result_date, ['pta'])

sub_id_order = [4, 5, 6, 7, 8, 12, 13, 0, 1, 2, 3, 9, 10, 11, 14]
# print('Result of each subject')
# for subject_id in sub_id_order:
#     NRMSE, _ = reader_cnn.get_one_trial_NRMSE_mean_std(sub_id_list=[subject_id])
#     MAE, _ = reader_cnn.get_param_mean_std_of_trial_mean('absolute mean error', ['All trials'], sub_id_list=[subject_id])
#
#     pearson_cor, _ = reader_cnn.get_param_mean_std_of_trial_mean('pearson correlation', ['All trials'], sub_id_list=[subject_id])
#     pearson_cor_pta, _ = reader_pta.get_param_mean_std_of_trial_mean('pearson correlation', ['All trials'], sub_id_list=[subject_id])
#
#     # print(SUB_NAMES[subject_id], end='\t')
#     print('{:.1f}'.format(round(NRMSE, 1)), end='\t')
#     print('{:.1f}'.format(round(MAE, 1)), end='\t')
#     print('{:.2f}'.format(round(pearson_cor, 2)), end='\t')
#     print('', end='\t')
#     print('{:.2f}'.format(round(pearson_cor_pta, 2)))


print('\n\n\nResult of each gait:')

# RUNNING_TRIALS = ('nike baseline 24', 'nike SI 24', 'nike SR 24', 'nike baseline 28', 'nike SI 28', 'nike SR 28',
#                   'mini baseline 24', 'mini SI 24', 'mini SR 24', 'mini baseline 28', 'mini SI 28', 'mini SR 28')

for trial_list in [_24_TRIALS, _28_TRIALS, NIKE_TRIALS, MINI_TRIALS]:
    # NRMSE, _ = reader_cnn.get_one_trial_NRMSE_mean_std(trial_name=trial_list)
    MAE, _ = reader_cnn.get_param_mean_std_of_trial_mean('absolute mean error', trial_list)
    pearson_cor, _ = reader_cnn.get_param_mean_std_of_trial_mean('pearson correlation', trial_list)
    pearson_cor_pta, _ = reader_pta.get_param_mean_std_of_trial_mean('pearson correlation', trial_list)
    # print('{:.1f}'.format(round(NRMSE, 1)), end='\t')
    print('{:.1f}'.format(round(MAE, 1)), end='\t')
    print('{:.2f}'.format(round(pearson_cor, 2)), end='\t')
    print('', end='\t')
    print('{:.2f}'.format(round(pearson_cor_pta, 2)))

print('step result')
for trial_list in [_24_TRIALS, _28_TRIALS, NIKE_TRIALS, MINI_TRIALS]:
    pearson_cor, _, MAE, _ = reader_cnn.get_param_mean_std_of_all_steps(trial_list, sub_id_order)
    NRMSE, _ = reader_cnn.get_NRMSE_mean_std_of_all_steps(trial_list, sub_id_order)
    pearson_cor_pta, _, _, _ = reader_pta.get_param_mean_std_of_all_steps(trial_list, sub_id_order)

    print('{:.1f}'.format(round(NRMSE, 1)), end='\t')
    print('{:.1f}'.format(round(MAE, 1)), end='\t')
    print('{:.2f}'.format(round(pearson_cor, 2)), end='\t')
    print('', end='\t')
    print('{:.2f}'.format(round(pearson_cor_pta, 2)))





