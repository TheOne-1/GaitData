
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from const import LINE_WIDTH, FONT_DICT, FONT_DICT_SMALL
from plot_funcs import format_subplot

# load data
example_data_path = 'D:\Tian\Research\Projects\HuaweiProject\SharedDocs\Huawei\PhaseIData\ProcessedData\\' \
                    '190521GongChangyang\\200Hz\\'
gait_data_df = pd.read_csv(example_data_path + 'nike baseline 24.csv', index_col=False)
gait_param_file = pd.read_csv(example_data_path + 'param_of_nike baseline 24.csv', index_col=False)

# set filter
wn = 10 / 200
b = firwin(100, wn)

l_foot_acc_z = -gait_data_df['l_foot_acc_z'].values
l_foot_acc_z = lfilter(b, 1, l_foot_acc_z)
l_strikes = gait_param_file['strikes_IMU_lfilter'].values
l_strike_indexes = np.where(l_strikes == 1)[0]

l_strikes_real = gait_param_file['l_strikes'].values
l_strike_indexes_real = np.where(l_strikes_real == 1)[0] + 50

plt.figure(figsize=(9, 6))
plt.plot(l_foot_acc_z, linewidth=2)
esti_plot, = plt.plot(l_strike_indexes, l_foot_acc_z[l_strike_indexes], 'g*', markersize=12)
real_plot, = plt.plot(l_strike_indexes_real, l_foot_acc_z[l_strike_indexes_real], 'ro', markersize=7)
format_subplot()

plt.legend([esti_plot, real_plot], ['estimated strikes', 'real strikes'], fontsize=16)
plt.xlabel('time', fontdict=FONT_DICT_SMALL)
plt.ylabel('vertical acceleration', fontdict=FONT_DICT_SMALL)
ax = plt.gca()
start_sample = 8520
ax.set_xlim(start_sample, start_sample + 400)
ax.set_xticks(range(start_sample, start_sample + 401, 100))
ax.set_xticklabels(['0s', '0.5s', '1s', '1.5s', '2s'], fontdict=FONT_DICT_SMALL)

ax.set_ylim(-24, 23)
ax.set_yticks(range(-20, 23, 10))
ax.set_yticklabels(range(-20, 23, 10), fontdict=FONT_DICT_SMALL)
plt.savefig('exported_figures/strikes.png')
plt.show()



