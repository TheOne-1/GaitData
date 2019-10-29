"""
Figure is not a good choice, changed to a table showing that one IMU or five IMUs gives the same result.
"""

import numpy as np
from Drawer import Drawer, ResultReader, ComboResultReader
from ComboGenerator import ComboGenerator
import copy


segments = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
result_date = '1026'
precision = 3

mean_array, std_array = np.zeros([3, 5]), np.zeros([3, 5])

c51_combo_reader = ComboResultReader(result_date, [[segment] for segment in segments])
mean_array[:, 0], std_array[:, 0], best_combo = c51_combo_reader.get_combo_best_mean_std()
print(best_combo, end=' ')

c52_combos = ComboGenerator.combinations_by_subset(segments, 2)
c52_combo_reader = ComboResultReader(result_date, c52_combos)
mean_array[:, 1], std_array[:, 1], best_combo = c52_combo_reader.get_combo_best_mean_std()
print(best_combo, end=' ')

c53_combos = ComboGenerator.combinations_by_subset(segments, 3)
c53_combo_reader = ComboResultReader(result_date, c53_combos)
mean_array[:, 2], std_array[:, 2], best_combo = c53_combo_reader.get_combo_best_mean_std()
print(best_combo, end=' ')

c54_combos = [copy.deepcopy(segments) for i in range(5)]
for combo, segment in zip(c54_combos, segments):
    combo.remove(segment)
c54_combo_reader = ComboResultReader(result_date, c54_combos)
mean_array[:, 3], std_array[:, 3], best_combo = c54_combo_reader.get_combo_best_mean_std()
print(best_combo, end=' ')

c55_combo_reader = ComboResultReader(result_date, [segments])
mean_array[:, 4], std_array[:, 4], _ = c55_combo_reader.get_combo_best_mean_std()
print()

# mean_array, std_array = np.round(mean_array, precision), np.round(std_array, precision)
# print the result
for i_row in range(3):
    print()
    for i_col in range(5):
        if i_row == 0:
            print(str(round(mean_array[i_row, i_col], 2)) + '(' + str(round(std_array[i_row, i_col], 2)) + ')', end='\t')
            # print(str(round(mean_array[i_row, i_col], 3)), end='\t')
        else:
            print(str(round(mean_array[i_row, i_col], 1)) + '(' + str(round(std_array[i_row, i_col], 1)) + ')', end='\t')


# print the result of PTA model

c55_combo_reader = ComboResultReader(result_date, [['pta']])
mean_pta, std_pta, _ = c55_combo_reader.get_combo_best_mean_std()
print('\n' + str(mean_pta[0])[:5])


