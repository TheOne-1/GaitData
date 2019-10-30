from const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES, FONT_SIZE, FONT_SIZE_SMALL, FONT_DICT, TRIAL_NAMES, \
    SI_SR_TRIALS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


class Drawer:
    @staticmethod
    def draw_one_imu_result(mean_values, std_values):
        plt.figure(figsize=(7, 6))
        bar_locs = [0, 3, 5, 7, 9, 1]
        Drawer.format_plot()
        bars, ebars = [], []
        for i_segment in range(5):
            bars.append(plt.bar(bar_locs[i_segment], mean_values[i_segment], color='grey', width=1,
                                label='CNN model'))
        bars.append(plt.bar(bar_locs[5], mean_values[5], color='white', edgecolor='black', hatch='//', width=1,
                            linewidth=LINE_WIDTH, label='PTA model'))

        plt.legend(handles=bars[-2:], bbox_to_anchor=[0.53, 0.88], ncol=1, fontsize=FONT_SIZE,
                   frameon=False, handlelength=2, handleheight=1.3)

        plt.plot([-1, 3], [0, 0], linewidth=LINE_WIDTH, color='black')
        ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_values, std_values,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True,
                                                   elinewidth=LINE_WIDTH)
        Drawer.format_errorbar_cap(caplines)
        plt.tight_layout(rect=[0.07, 0.06, 0.98, 0.98])
        Drawer.set_one_imu_ticks(bar_locs)
        plt.savefig('paper_figures/comparison of one IMU result.png')
        plt.show()

    @staticmethod
    def set_one_imu_ticks(bar_locs):
        ax = plt.gca()
        ax.set_xlim(-1, 10)
        ax.set_xticks([0.5, 3, 5, 7, 9])
        ax.set_xticklabels(['shank', 'foot', 'pelvis', 'trunk', 'thigh'], fontdict=FONT_DICT)
        ax.set_xlabel('IMU attachment location', labelpad=6, fontdict=FONT_DICT)

        ax.set_ylim(0, 1.05)
        y_range = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT)
        ax.set_ylabel('Pearson\'s correlation coefficient', labelpad=6, fontdict=FONT_DICT)

    @staticmethod
    def draw_compare_bars(true_mean_values, true_std_values, pred_mean_values, pred_std_values):
        plt.figure(figsize=(7, 6))
        Drawer.format_plot()
        bars, ebars = [], []
        for i_cate in range(4):
            bars.append(
                plt.bar(i_cate * 3, true_mean_values[i_cate], color='darkgray', width=0.8, label='VALR - force plate'))
            bars.append(plt.bar(i_cate * 3 + 1, pred_mean_values[i_cate], color='dimgray', width=0.8, label='VALR - CNN model'))

        plt.legend(handles=bars[:2], bbox_to_anchor=[0.63, 0.85], ncol=1, handlelength=2, handleheight=1.3,
                   fontsize=FONT_SIZE, frameon=False)

        plt.plot([-1, 3], [0, 0], linewidth=LINE_WIDTH, color='black')
        ebar, caplines, barlinecols = plt.errorbar(range(0, 12, 3), true_mean_values, true_std_values,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True,
                                                   elinewidth=LINE_WIDTH)
        Drawer.format_errorbar_cap(caplines)
        ebar, caplines, barlinecols = plt.errorbar(range(1, 12, 3), pred_mean_values, pred_std_values,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True,
                                                   elinewidth=LINE_WIDTH)
        Drawer.format_errorbar_cap(caplines)
        plt.tight_layout(rect=[0.07, 0.06, 0.98, 0.98])
        Drawer.set_compare_bar_ticks()
        plt.savefig('paper_figures/comparison bars.png')

    @staticmethod
    def set_compare_bar_ticks():
        ax = plt.gca()
        ax.set_xlim(-1, 11)
        ax.set_xticks(np.arange(0.5, 11, 3))
        ax.set_xticklabels(['2.4 m/s', '2.8 m/s', '2.4 m/s', '2.8 m/s'], fontdict=FONT_DICT)
        fig = plt.gcf()
        fig.text(0.17, 0.03, 'traditional footwear    minimalist footwear', fontdict=FONT_DICT)

        ax.set_ylim(0, 150)
        y_range = range(0, 151, 30)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT)
        ax.set_ylabel('Average VALR (BW/s)', labelpad=6, fontdict=FONT_DICT)

    @staticmethod
    def format_plot():
        mpl.rcParams['hatch.linewidth'] = LINE_WIDTH  # previous svg hatch linewidth
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_tick_params(width=LINE_WIDTH)
        ax.yaxis.set_tick_params(width=LINE_WIDTH)
        ax.spines['left'].set_linewidth(LINE_WIDTH)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH)

    @staticmethod
    def format_errorbar_cap(caplines):
        for i_cap in range(1):
            caplines[i_cap].set_marker('_')
            caplines[i_cap].set_markersize(12)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    @staticmethod
    def draw_example_result(true_lr_list, pred_lr_list, title):
        # colors = ['darkgreen', 'darkgreen', 'slategrey', 'slategrey']
        # patterns = ['^', '*', '^', '*']
        plt.figure(figsize=(7, 6))
        Drawer.format_plot()
        plt.title(title)
        plt.plot([0, 150], [0, 150], 'black')
        cate_num = len(true_lr_list)
        for i_cate in range(cate_num):
            plt.plot(true_lr_list[i_cate], pred_lr_list[i_cate], 'b.')
        plt.tight_layout(rect=[0.07, 0.06, 0.98, 0.98])
        Drawer.set_example_bar_ticks()
        plt.savefig('paper_figures/example result.png')

    @staticmethod
    def set_example_bar_ticks():
        ax = plt.gca()
        ax.set_xlim(0, 160)
        ax.set_xticks(range(0, 151, 30))
        ax.set_xticklabels(range(0, 151, 30), fontdict=FONT_DICT)
        ax.set_xlabel('VALR - force plate (BW/s)', labelpad=6, fontdict=FONT_DICT)

        ax.set_ylim(0, 160)
        y_range = range(0, 151, 30)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT)
        ax.set_ylabel('VALR - CNN model (BW/s)', labelpad=6, fontdict=FONT_DICT)


class ResultReader:
    def __init__(self, date, segment_names):
        summary_file_path = '../2_loading_rate/result_conclusion/' + date + '/trial_summary/' + date
        for segment in segment_names:
            summary_file_path = summary_file_path + '_' + segment
        summary_file_path = summary_file_path + '.xlsx'
        self._result_df = pd.read_excel(summary_file_path)

        step_result_file_path = '../2_loading_rate/result_conclusion/' + date + '/step_result/' + date
        for segment in segment_names:
            step_result_file_path = step_result_file_path + '_' + segment
        step_result_file_path = step_result_file_path + '.csv'
        self._step_result_df = pd.read_csv(step_result_file_path)

    def get_param_mean_std(self, param_name, trial_list, sub_id_list=None):
        param_array = self.get_param_values(param_name, trial_list, sub_id_list)
        param_mean, param_std = np.mean(param_array), np.std(param_array)
        return param_mean, param_std

    def get_param_values(self, param_name, trial_list, sub_id_list=None):
        """

        :param param_name:
        :param trial_list: list of str, e.g. ['all trials'], MINI_SI_SR_TRIALS, NIKE_SI_SR_TRIALS, ...
        :param sub_id_list:
        :return:
        """
        param_df = self._result_df[self._result_df['parameter name'] == param_name]
        # remove the overall mean row
        param_df = param_df[param_df['subject name'] != 'absolute mean']
        if sub_id_list is not None:
            sub_name_list = [SUB_NAMES[sub_id] for sub_id in sub_id_list]
            param_df = param_df[param_df['subject name'].isin(sub_name_list)]
        param_array = param_df[trial_list].values
        return param_array

    def get_lr_values(self, sub_id_list, trial_id_list):
        for trial_id in trial_id_list:
            if TRIAL_NAMES[trial_id] not in SI_SR_TRIALS:
                raise ValueError('Wrong trial id. The trial must be a SI or a SR trial.')

        step_result_df = self._step_result_df
        step_result_df = step_result_df[step_result_df['subject id'].isin(sub_id_list)]
        step_result_df = step_result_df[step_result_df['trial id'].isin(trial_id_list)]
        true_lr = step_result_df['true LR']
        pred_lr = step_result_df['predicted LR']
        return true_lr, pred_lr

    def get_all_trial_NRMSE_mean_std(self, sub_id_list=None):
        trial_list = 'All trials'
        # step 0, get values
        param_df = self._result_df[self._result_df['parameter name'] == 'RMSE']
        # remove the overall mean row
        param_df = param_df[param_df['subject name'] != 'absolute mean']
        if sub_id_list is not None:
            sub_name_list = [SUB_NAMES[sub_id] for sub_id in sub_id_list]
            param_df = param_df[param_df['subject name'].isin(sub_name_list)]
        NRMSE_list = []
        for sub_name, RMSE in zip(param_df['subject name'], param_df[trial_list]):
            sub_id = SUB_NAMES.index(sub_name)
            sub_min, sub_max = self.get_sub_min_max_LR(sub_id)
            NRMSE_list.append(RMSE / (sub_max - sub_min) * 100)
        NRMSE_mean, NRMSE_std = np.mean(NRMSE_list), np.std(NRMSE_list)
        return NRMSE_mean, NRMSE_std

    def get_sub_min_max_LR(self, sub_id):
        step_result_df = self._step_result_df
        sub_df = step_result_df[step_result_df['subject id'] == sub_id]
        sub_true_LR_array = sub_df['true LR'].values
        return np.min(sub_true_LR_array), np.max(sub_true_LR_array)


class ComboResultReader:
    def __init__(self, result_date, segment_combos):
        self.result_date = result_date
        self.segment_combos = segment_combos

    def get_combo_best_mean_std(self):
        # case c51
        combo_best_mean, combo_best_std = np.full([3], -1.0), np.full([3], -1.0)
        best_combo = None
        for combo in self.segment_combos:
            result_reader = ResultReader(self.result_date, combo)
            mean_NRMSE, _ = result_reader.get_all_trial_NRMSE_mean_std()
            if combo_best_mean[0] == -1 or mean_NRMSE < combo_best_mean[1]:
                combo_best_mean[0], combo_best_std[0] = result_reader.get_param_mean_std('pearson correlation', ['All trials'])
                combo_best_mean[1], combo_best_std[1] = result_reader.get_all_trial_NRMSE_mean_std()
                combo_best_mean[2], combo_best_std[2] = result_reader.get_param_mean_std('absolute mean error',
                                                                                         ['All trials'])
                best_combo = combo
        return combo_best_mean, combo_best_std, best_combo
