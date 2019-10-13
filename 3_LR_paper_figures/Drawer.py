from const import LINE_WIDTH, FONT_DICT_SMALL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Drawer:

    @staticmethod
    def draw_best_combos(mean_array, std_array):
        plt.figure(figsize=(9, 6))
        Drawer.format_plot()
        bars, ebars = [], []
        for i_cate in range(5):
            bars.append(plt.bar(i_cate, mean_array[i_cate], color='gray', width=0.7))

        plt.plot([-1, 3], [0, 0], linewidth=LINE_WIDTH, color='black')
        ebar, caplines, barlinecols = plt.errorbar(range(5), mean_array, std_array,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True, uplims=True,
                                                   elinewidth=LINE_WIDTH)
        for i_cap in range(2):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(14)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)
        Drawer.set_best_combo_ticks()
        # plt.savefig('fpa_figures/fpa error of speeds.png')
        plt.show()

    @staticmethod
    def set_best_combo_ticks():
        ax = plt.gca()
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks(np.arange(0, 3, 1))
        ax.set_xticklabels(['1.0 m/s', '1.2 m/s', '1.4 m/s'], fontdict=FONT_DICT_SMALL)

        ax.set_ylim(-2.5, 2.5)
        y_range = range(-2, 3, 1)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('Average FPA error (deg)', labelpad=10, fontdict=FONT_DICT_SMALL)

    @staticmethod
    def format_plot():
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=LINE_WIDTH)
        ax.yaxis.set_tick_params(width=LINE_WIDTH)
        ax.spines['left'].set_linewidth(LINE_WIDTH)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH)

    @staticmethod
    def format_axis():
        plt.plot([-15, 55], [-15, 55], 'r-')
        ax = plt.gca()
        ax.set_xlim(-20, 65)
        ax.set_xticks(range(-15, 65, 15))
        ax.set_xticklabels(range(-15, 65, 15), fontdict=FONT_DICT_SMALL)
        ax.set_ylim(-20, 65)
        ax.set_yticks(range(-15, 65, 15))
        ax.set_yticklabels(range(-15, 65, 15), fontdict=FONT_DICT_SMALL)
        plt.xlabel('true value (degree)', fontdict=FONT_DICT_SMALL)
        plt.ylabel('predicted value (degree)', fontdict=FONT_DICT_SMALL)


class ResultReader:
    def __init__(self, date, segment_names):
        file_path = '../2_loading_rate/result_conclusion/' + date + '/trial_summary/' + date
        for segment in segment_names:
            file_path = file_path + '_' + segment
        file_path = file_path + '.xlsx'
        self._result_df = pd.read_excel(file_path)

    def get_param_mean_std(self, param_name, trial_list):
        """

        :param param_name:
        :param trial_list: list of str, e.g. ['all trials'], MINI_SI_SR_TRIALS, NIKE_SI_SR_TRIALS, ...
        :return:
        """

        param_df = self._result_df[self._result_df['parameter name'] == param_name]
        # remove the overall mean row
        param_df = param_df[param_df['subject name'] != 'absolute mean']
        param_df = param_df[trial_list].values
        param_mean, param_std = np.mean(param_df), np.std(param_df)
        return param_mean, param_std


class ComboResultReader:
    def __init__(self, result_date, segment_combos):
        self.result_date = result_date
        self.segment_combos = segment_combos

    def get_combo_best_mean_std(self):
        # case c51
        combo_best_mean, combo_best_std = 0, 0
        for segment in self.segment_combos:
            result_reader = ResultReader(self.result_date, segment)
            combo_mean, combo_std = result_reader.get_param_mean_std('pearson correlation', ['All trials'])
            if combo_mean > combo_best_mean:
                combo_best_mean, combo_best_std = combo_mean, combo_std

        return combo_best_mean, combo_best_std

