import pandas as pd
import matplotlib.pyplot as plt
from Drawer import Drawer
import numpy as np

neurons_tested = [
    [2, 2, 2],
    [4, 4, 4],
    [8, 8, 8],
    [16, 16, 10],
    [32, 32, 10],
    [50, 50, 10],
    [64 for _ in range(2)] + [10],
    [128 for _ in range(2)] + [10],
    [256 for _ in range(2)] + [10],
    [512 for _ in range(2)] + [10],
    [1024 for _ in range(2)] + [10]
]

date = '1037'
# param_name = 'absolute mean error'
param_name = 'pearson correlation'
file_path = 'result_conclusion/' + date + '/layer_summary.csv'

result_data_df = pd.read_csv(file_path, index_col=False)
param_df = result_data_df[param_name]

layer_nums = [1, 2, 3, 4, 8, 16, 32, 64]
layer_num_accuracy = param_df.values[1:len(layer_nums)+1]
figure_param_dict = {'marker_label': 'Layer Number of the Proposed CNN',
                     'x_label': 'Hidden Layer Number',
                     'x_ticks': [2 ** x for x in range(0, 7)],
                     'x_tick_labels': ['$2^{:.0f}$'.format(x) for x in range(0, 7)],
                     'used_loc': 2}
Drawer.show_hidden_layer(layer_nums, layer_num_accuracy, param_name, figure_param_dict)

neuron_nums = [x[0] for x in neurons_tested]
layer_num_accuracy = []
for neurons in neurons_tested:
    layer_name = 'layer_' + str(neurons[0]) + '_' + str(neurons[1]) + '_' + str(neurons[2])
    layer_df = result_data_df[result_data_df['layer_name'] == layer_name]
    layer_num_accuracy.append(layer_df[param_name].values[0])

figure_param_dict = {'marker_label': 'Neuron Number of the Proposed CNN',
                     'x_label': 'Neuron Number of the First Two Layers',
                     'x_ticks': [2 ** x for x in range(1, 11)],
                     'x_tick_labels': ['$2^{' + str(x) + '}$' for x in range(1, 11)],
                     'used_loc': 5}
Drawer.show_hidden_layer(neuron_nums, layer_num_accuracy, param_name, figure_param_dict)

plt.show()
