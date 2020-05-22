"""
Comparison between five segments and PTA
"""

import numpy as np
from Drawer import Drawer, ResultReader, ComboResultReader
from scipy.stats import ttest_ind
import statsmodels.stats.multicomp as ml

segments = ['l_shank', 'l_foot', 'pelvis', 'trunk', 'l_thigh']
result_date = '1028'
precision = 3
param_name = 'pearson correlation'

mean_array, std_array = np.zeros([6]), np.zeros([6])
param_values_list = []
# load results of five segments
for i_segment in range(5):
    segment = segments[i_segment]
    segment_reader = ResultReader(result_date, [segment])
    param_values = segment_reader.get_param_values(param_name, 'All trials')
    param_values_list.append(param_values)
    mean_array[i_segment], std_array[i_segment] = np.mean(param_values), np.std(param_values)

# load results of the PTA model
pta_reader = ResultReader(result_date, ['pta'])
param_values = pta_reader.get_param_values(param_name, 'All trials')
param_values_list.append(param_values)
mean_array[5], std_array[5] = np.mean(param_values), np.std(param_values)

# print ANOVA test result
_, pvalue = ttest_ind(param_values_list[0], param_values_list[5])
print('PTA and CNN pvalue: ' + str(round(pvalue, 5)), end='\n\n')

# calculate Cohen's d
mean_diff = mean_array[0] - mean_array[5]
pooled_values = np.concatenate([param_values_list[0], param_values_list[5]])
pooled_std = np.std(param_values)
print('Mean difference: ' + str(mean_diff)[:4] + '  Pooled std: ' + str(pooled_std)[:4] +
      '  Cohen\'s d: ' + str(mean_diff / pooled_std)[:4] + '\n')

data = np.zeros([0])
groups = np.zeros([0])
for i_segment in range(5):
    data = np.concatenate([data, param_values_list[i_segment]])
    groups = np.concatenate([groups, np.full(param_values_list[i_segment].shape, i_segment)])
mcobj = ml.MultiComparison(data, groups)
out = mcobj.tukeyhsd(0.05)
print(out)

mean_array, std_array = Drawer.add_extra_correlation_from_citation(mean_array, std_array)
Drawer.draw_one_imu_result(mean_array, std_array)

