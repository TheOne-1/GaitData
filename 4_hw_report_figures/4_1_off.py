
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

l_foot_gyr_x = -gait_data_df['l_foot_gyr_x'].values
l_foot_gyr_x = lfilter(b, 1, l_foot_gyr_x)
l_offs = gait_param_file['offs_IMU_lfilter'].values
l_off_indexes = np.where(l_offs == 1)[0]

l_offs_real = gait_param_file['l_offs'].values
l_off_indexes_real = np.where(l_offs_real == 1)[0][:-1] + 50

plt.figure(figsize=(9, 6))
plt.plot(l_foot_gyr_x, linewidth=2)
esti_plot, = plt.plot(l_off_indexes, l_foot_gyr_x[l_off_indexes], 'g*', markersize=12)
real_plot, = plt.plot(l_off_indexes_real, l_foot_gyr_x[l_off_indexes_real], 'ro', markersize=7)
format_subplot()

plt.legend([esti_plot, real_plot], ['estimated offs', 'real offs'], fontsize=16)
plt.xlabel('time', fontdict=FONT_DICT_SMALL)
plt.ylabel('pitch angular velocity', fontdict=FONT_DICT_SMALL)
ax = plt.gca()
ax.set_xlim(4500, 4900)
ax.set_xticks(range(4500, 4901, 100))
ax.set_xticklabels(['0s', '0.5s', '1s', '1.5s', '2s'], fontdict=FONT_DICT_SMALL)

ax.set_ylim(-9, 12)
ax.set_yticks(range(-8, 13, 4))
ax.set_yticklabels(range(-8, 13, 4), fontdict=FONT_DICT_SMALL)
plt.savefig('exported_figures/offs.png')
plt.show()



