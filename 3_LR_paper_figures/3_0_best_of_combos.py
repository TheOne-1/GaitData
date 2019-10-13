import numpy as np
import pandas as pd
from Drawer import Drawer, ResultReader, ComboResultReader
from ComboGenerator import ComboGenerator
import matplotlib.pyplot as plt


segments = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
result_date = '1010'

mean_array, std_array = np.zeros([5]), np.zeros([5])

c51_combo_reader = ComboResultReader(result_date, [[segment] for segment in segments])
mean_array[0], std_array[0] = c51_combo_reader.get_combo_best_mean_std()

c52_combos = ComboGenerator.combinations_by_subset(segments, 2)
c52_combo_reader = ComboResultReader(result_date, c52_combos)
mean_array[1], std_array[1] = c52_combo_reader.get_combo_best_mean_std()

Drawer.draw_best_combos(mean_array, std_array)


# if c53:
#     print('\n\nDoing C53')
#     segment_combos = ComboGenerator.combinations_by_subset(segments, 3)
#     for combo in segment_combos:
#         print('\nCurrent segments: ' + str(combo))
#         cross_vali_LR_processor = ProcessorLR(train, {}, combo)
#         test_name = date
#         for segment in combo:
#             test_name = test_name + '_' + segment
#         cross_vali_LR_processor.cnn_cross_vali(test_name=test_name, plot=False)
#         keras.backend.clear_session()
#
# if c54:
#     print('\n\nDoing C54')
#     for segment in segments[2:]:
#         segment_list = copy.deepcopy(segments)
#         segment_list.remove(segment)
#         print('\nCurrent segments: ' + str(segment_list))
#         cross_vali_LR_processor = ProcessorLR(train, {}, segment_list)
#         test_name = date
#         for segment in segment_list:
#             test_name = test_name + '_' + segment
#         cross_vali_LR_processor.cnn_cross_vali(test_name=test_name, plot=False)
#         keras.backend.clear_session()
#
# if c55:
#     print('\n\nDoing all segment')
#     cross_vali_LR_processor = ProcessorLR(train, {}, segments)
#     test_name = date
#     for segment in segments:
#         test_name = test_name + '_' + segment
#     cross_vali_LR_processor.cnn_cross_vali(test_name=test_name, plot=False)
#     keras.backend.clear_session()



