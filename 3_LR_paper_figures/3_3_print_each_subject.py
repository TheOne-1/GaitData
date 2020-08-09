"""
Corresponding to table I and II
"""

from Drawer import Drawer, ResultReader
from const import SUB_NAMES, TRIAL_NAMES, _24_TRIALS, _28_TRIALS, NIKE_TRIALS, MINI_TRIALS, SUB_ID_PAPER
from scipy.stats import ttest_ind

result_date = '1028'
reader_cnn = ResultReader(result_date, ['l_shank'])
reader_pta = ResultReader(result_date, ['pta'])
sub_info_df = ResultReader.get_subject_info()

print('Result of each subject')
for subject_id in SUB_ID_PAPER:
    # NRMSE, _ = reader_cnn.get_one_trial_NRMSE_mean_std(sub_id_list=[subject_id])
    # MAE, _ = reader_cnn.get_param_mean_std_of_trial_mean('absolute mean error', ['All trials'], sub_id_list=[subject_id])
    # print('{:.1f}'.format(round(NRMSE, 1)), end='\t')
    # print('{:.1f}'.format(round(MAE, 1)), end='\t')

    pearson_cor, _ = reader_cnn.get_param_mean_std_of_trial_mean('pearson correlation', ['All trials'],
                                                                 sub_id_list=[subject_id])
    pearson_cor_pta, _ = reader_pta.get_param_mean_std_of_trial_mean('pearson correlation', ['All trials'],
                                                                     sub_id_list=[subject_id])

    print(str(sub_info_df.iloc[subject_id, 3]), end='\t')
    print('{:.2f}'.format(round(sub_info_df.iloc[subject_id, 5] / 100, 2)), end='\t')
    print(str(sub_info_df.iloc[subject_id, 4]), end='\n')
    # print('{:.2f}'.format(round(pearson_cor, 2)), end='\t')
    # print('{:.2f}'.format(round(pearson_cor_pta, 2)))

print('\n\n\nResult of each gait:')

# for trial_list in [_24_TRIALS, _28_TRIALS, NIKE_TRIALS, MINI_TRIALS]:
#     MAE, _ = reader_cnn.get_param_mean_std_of_trial_mean('absolute mean error', trial_list)
#     pearson_cor, _ = reader_cnn.get_param_mean_std_of_trial_mean('pearson correlation', trial_list)
#     pearson_cor_pta, _ = reader_pta.get_param_mean_std_of_trial_mean('pearson correlation', trial_list)
#     print('{:.1f}'.format(round(MAE, 1)), end='\t')
#     print('{:.2f}'.format(round(pearson_cor, 2)), end='\t')
#     print('', end='\t')
#     print('{:.2f}'.format(round(pearson_cor_pta, 2)))

print('step result')
for trial_list in [_24_TRIALS, _28_TRIALS, NIKE_TRIALS, MINI_TRIALS]:
    trial_id_list = [float(TRIAL_NAMES.index(trial_name)) for trial_name in trial_list]

    pearson_cor, pearson_cor_std, pearson_cor_values, _, _ = reader_cnn.get_param_mean_std_of_all_steps(
        SUB_ID_PAPER, trial_id_list)
    pearson_cor_pta, pearson_cor_pta_std, pearson_cor_pta_values, _, _ = reader_pta.get_param_mean_std_of_all_steps(
        SUB_ID_PAPER, trial_id_list)

    print('{:.2} ({:.2})'.format(round(pearson_cor, 2), round(pearson_cor_std, 2)), end='\t\t')
    print('{:.2} ({:.2})'.format(round(pearson_cor_pta, 2), round(pearson_cor_pta_std, 2)), end='\t\t')
    _, pvalue = ttest_ind(pearson_cor_values, pearson_cor_pta_values)
    print('%.0e' % pvalue)
    # print('< 0.05')

# read the step type file
for trial_type in ['forefoot', 'midfoot', 'rearfoot', 'low_rate', 'mid_rate', 'high_rate']:
    pearson_cor, pearson_cor_std, pearson_cor_values, _, _ = reader_cnn.get_param_mean_std_of_all_steps(
        SUB_ID_PAPER, [trial_type], select_col_name='step type')
    pearson_cor_pta, pearson_cor_pta_std, pearson_cor_pta_values, _, _ = reader_pta.get_param_mean_std_of_all_steps(
        SUB_ID_PAPER, [trial_type], select_col_name='step type')

    print('{:.2} ({:.2})'.format(round(pearson_cor, 2), round(pearson_cor_std, 2)), end='\t\t')
    print('{:.2} ({:.2})'.format(round(pearson_cor_pta, 2), round(pearson_cor_pta_std, 2)), end='\t\t')
    _, pvalue = ttest_ind(pearson_cor_values, pearson_cor_pta_values)
    # print(pvalue)
    print('%.0e' % pvalue)
