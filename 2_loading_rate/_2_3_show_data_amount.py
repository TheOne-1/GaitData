import pandas as pd
import matplotlib.pyplot as plt
from Drawer import Drawer
import numpy as np

optimal_segment_list = [
    ['l_shank'],
    ['l_shank', 'l_foot'],
    ['l_thigh', 'l_shank', 'l_foot'],
    ['trunk', 'pelvis', 'l_shank', 'l_foot'],
    ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
]

percents = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]

date = '1032'

param_name = 'train_loss'
# param_name = 'validation_loss'
file_path = 'result_conclusion/' + date + '/loss_summary.csv'

result_data_df = pd.read_csv(file_path, index_col=False)
Drawer.show_data_amount(result_data_df, optimal_segment_list, percents, param_name)

plt.show()
