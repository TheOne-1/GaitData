import pandas as pd
import matplotlib.pyplot as plt
from Drawer import Drawer


optimal_segment_list = [
    ['l_shank'],
    ['l_shank', 'l_foot'],
    ['l_thigh', 'l_shank', 'l_foot'],
    ['trunk', 'pelvis', 'l_shank', 'l_foot'],
    ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
]
percents = [3**-4, 3**-3, 3**-2, 3**-1, 0.5, 0.7, 3**0]

date = '1032'
param_name = 'pearson correlation'

file_path = 'result_conclusion/' + date + '/amount_summary.csv'
result_data_df = pd.read_csv(file_path, index_col=False)

plt.figure()

for segment_list in optimal_segment_list:
    test_name = segment_list[0]
    for segment in segment_list[1:]:
        test_name = test_name + '_' + segment

    current_result_df = result_data_df[result_data_df['segment'] == test_name]
    plt.plot(percents, current_result_df[param_name])

plt.show()
