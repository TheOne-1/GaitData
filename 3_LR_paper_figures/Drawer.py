from const import LINE_WIDTH, FONT_DICT_SMALL, SUB_NAMES, FONT_SIZE, FONT_SIZE_SMALL, FONT_DICT, TRIAL_NAMES, \
    SI_SR_TRIALS, FONT_DICT_LONG_FIG, FONT_SIZE_LONG_FIG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines
from Evaluation import Evaluation


class Drawer:
    @staticmethod
    def draw_clip_sensitivity(the_range, the_accuracy, color='g', name='start', used_loc=3):
        plt.figure(figsize=(10, 7))
        time = [x * 5 for x in the_range]
        fig_accuracy, = plt.plot(time, the_accuracy, linewidth=LINE_WIDTH, label='accuracy curve')
        fig_used_loc, = plt.plot(time[used_loc], the_accuracy[used_loc], color+'o', markersize=15, label='used window ' + name)
        # plt.plot([0, 0], [0.5, 1], '--', linewidth=LINE_WIDTH, color='gray')

        ax = plt.gca()
        ax.set_xticks(time)
        ax.set_xticklabels(time, fontdict=FONT_DICT_SMALL)
        if 'start' in name:
            x_label = 'Window start time before foot-strike (ms)'
        else:
            x_label = 'Window end time after foot-strike (ms)'
        ax.set_xlabel(x_label, labelpad=10, fontdict=FONT_DICT_SMALL)
        ax.set_ylim(0.5, 1)
        y_range = [x/10 for x in range(5, 11)]
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('Correlation coefficient', labelpad=10, fontdict=FONT_DICT_SMALL)
        Drawer.format_plot()

        plt.legend(handles=[fig_accuracy, fig_used_loc], bbox_to_anchor=[0.55, 0.95], ncol=1,
                   fontsize=FONT_SIZE_LONG_FIG, frameon=False, handlelength=2.5, handleheight=1.6)
        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

    @staticmethod
    def draw_one_imu_result(mean_values, std_values):

        bar_patterns = ['/', '\\', 'x', '.']
        bar_labels = ['PTA (Zhang et al. [22])', 'PTA (present study)', 'PTA (Laughton et al. [23])',
                      'PTA (Greenhalgh et al. [24])']
        plt.figure(figsize=(17, 8))
        bar_locs = [-1, 6.5, 8, 9.5, 11, 0.5, 2, 3.5, 5]
        Drawer.format_plot()
        bars, ebars = [], []
        for i_segment in range(5):
            bars.append(plt.bar(bar_locs[i_segment], mean_values[i_segment], color='gray', width=0.8,
                                label='Proposed CNN model'))
        for i_extra in range(5, 9):
            bars.append(plt.bar(bar_locs[i_extra], mean_values[i_extra], color='white', edgecolor='black',
                                hatch=bar_patterns[i_extra - 5], width=0.8, linewidth=LINE_WIDTH,
                                label=bar_labels[i_extra - 5]))

        plt.legend(handles=bars[4:], bbox_to_anchor=[0.99, 1.25], ncol=3, fontsize=FONT_SIZE_LONG_FIG,
                   frameon=False, handlelength=2.5, handleheight=1.6)

        plt.plot([-1, 3], [0, 0], linewidth=LINE_WIDTH, color='black')
        ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_values, std_values,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True,
                                                   elinewidth=LINE_WIDTH)
        Drawer.format_errorbar_cap(caplines)
        plt.tight_layout(rect=[-0.05, 0.06, 0.99, 1.03])
        Drawer.set_one_imu_ticks(bar_locs)
        plt.savefig('paper_figures/comparison of one IMU result.jpg')
        plt.show()

    @staticmethod
    def set_one_imu_ticks(bar_locs):
        ax = plt.gca()
        ax.set_xlim(-1.7, 11.5)
        ax.set_xticks(bar_locs)
        ax.set_xticklabels(['shank', 'foot', 'pelvis', 'trunk', 'thigh', 'shank', 'shank', 'shank', 'shank'],
                           fontdict=FONT_DICT_LONG_FIG)
        ax.set_xlabel('IMU Location', labelpad=10, fontdict=FONT_DICT_LONG_FIG)

        ax.set_ylim(0, 1.05)
        y_range = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_LONG_FIG)
        ax.set_ylabel('Correlation Coefficient', labelpad=10, fontdict=FONT_DICT_LONG_FIG)

    @staticmethod
    def add_extra_correlation_from_citation(mean_array, std_array):
        """add additional mean and std from citation"""
        mean_additional = []
        std_additional = []

        # citation 1: Comparison of the correlations between impact loading rates and peak accelerations measured at
        # two different body sites: Intra- and inter-subject analysis
        values_of_research_1 = [0.546, 0.793, 0.778, 0.913, 0.580, 0.495, 0.556, 0.802, 0.638, 0.486]
        mean_additional.append(np.mean(values_of_research_1))
        std_additional.append(np.std(values_of_research_1))

        # append the PTA obtained from this research
        mean_additional.append(mean_array[5])
        std_additional.append(std_array[5])
        mean_array = np.delete(mean_array, 5)
        std_array = np.delete(std_array, 5)

        # citation 2: Effect of Strike Pattern and Orthotic Intervention on Tibial Shock During Running, RFS
        mean_additional.append(0.585)
        std_additional.append(0)

        # citation 2: Effect of Strike Pattern and Orthotic Intervention on Tibial Shock During Running
        mean_additional.append(0.439)
        std_additional.append(0)

        mean_array = np.concatenate([mean_array, mean_additional])
        std_array = np.concatenate([std_array, std_additional])

        return mean_array, std_array

    @staticmethod
    def draw_compare_bars(true_mean_values, true_std_values, pred_mean_values, pred_std_values):
        fig = plt.figure(figsize=(11, 9))
        ax = plt.gca()
        ax.set_position([0.14, 0.12, 0.82, 0.7])

        Drawer.format_plot()
        bars, ebars = [], []
        for i_cate in range(4):
            bars.append(
                plt.bar(i_cate * 3, true_mean_values[i_cate], color='darkgray', width=0.8,
                        label='VALR: Laboratory Force Plate'))
            bars.append(
                plt.bar(i_cate * 3 + 1, pred_mean_values[i_cate], color='dimgray', width=0.8,
                        label='VALR: Single Shank IMU (Proposed CNN Model)'))

        plt.legend(handles=bars[:2], bbox_to_anchor=[1.03, 1.3], ncol=1, handlelength=2, handleheight=1.3,
                   fontsize=FONT_SIZE, frameon=False)

        plt.plot([-1, 3], [0, 0], linewidth=LINE_WIDTH, color='black')
        ebar, caplines, barlinecols = plt.errorbar(range(0, 12, 3), true_mean_values, true_std_values,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True,
                                                   elinewidth=LINE_WIDTH)
        Drawer.format_errorbar_cap(caplines)
        ebar, caplines, barlinecols = plt.errorbar(range(1, 12, 3), pred_mean_values, pred_std_values,
                                                   capsize=0, ecolor='black', fmt='none', lolims=True,
                                                   elinewidth=LINE_WIDTH)

        # new clear axis overlay with 0-1 limits
        l2 = lines.Line2D([0.55, 0.55], [0.01, 0.845], linestyle='--', transform=fig.transFigure, color='gray')
        fig.lines.extend([l2])

        # plt.plot([5, 5], [-20, 150], '--', color='grey')
        Drawer.format_errorbar_cap(caplines)
        Drawer.set_compare_bar_ticks()
        plt.savefig('paper_figures/comparison bars.jpg')

    @staticmethod
    def set_compare_bar_ticks():
        ax = plt.gca()
        ax.set_xlim(-1, 11)
        ax.set_xticks(np.arange(0.5, 11, 3))
        ax.set_xticklabels(['2.4 m·s${^{-1}}$', '2.8 m·s${^{-1}}$', '2.4 m·s${^{-1}}$', '2.8 m·s${^{-1}}$'],
                           fontdict=FONT_DICT)
        fig = plt.gcf()
        fig.text(0.204, 0.02, 'Standard Shoes           Minimalist Shoes', fontdict=FONT_DICT)

        ax.set_ylim(0, 150)
        y_range = range(0, 151, 30)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT)
        ax.set_ylabel('VALR (BW·s${^{-1}}$)', labelpad=6, fontdict=FONT_DICT)

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
            caplines[i_cap].set_markersize(25)
            caplines[i_cap].set_markeredgewidth(LINE_WIDTH)

    @staticmethod
    def draw_example_result(true_lr_list, pred_lr_list, title):
        colors = ['darkgreen', 'darkgreen', 'maroon', 'maroon']
        markers = ['^', 'o', '^', 'o']
        labels = ['Standard Shoes, 2.4 m/s', 'Standard Shoes, 2.8 m/s',
                  'Minimalist Shoes, 2.4 m/s', 'Minimalist Shoes, 2.8 m/s']
        transparency = [0.7, 0.7, 0.4, 0.4]
        plt.figure(figsize=(11, 9))
        Drawer.format_plot()
        plt.title(title)
        # plt.plot([0, 250], [0, 250], 'black')
        cate_num = len(true_lr_list)
        scatters = []
        for i_cate in range(cate_num):
            scatters.append(plt.scatter(true_lr_list[i_cate], pred_lr_list[i_cate], s=200, color=colors[int(i_cate / 2)],
                                        marker=markers[int(i_cate / 2)], alpha=transparency[int(i_cate/2)],
                                        label=labels[int(i_cate/2)]))
        Drawer.set_example_bar_ticks()
        plt.legend(handles=scatters[::2], bbox_to_anchor=[0.64, 1.04], ncol=1, fontsize=FONT_SIZE_SMALL,
                   frameon=False)

        plt.tight_layout(rect=[0, 0, 0.98, 0.98])
        plt.savefig('paper_figures/example result.png')

    @staticmethod
    def draw_example_result_one_cate(true_lr_list, pred_lr_list, title):
        """
        Only one color and one type of marker was used.
        :param true_lr_list:
        :param pred_lr_list:
        :param title:
        :return:
        """
        plt.figure(figsize=(11, 9))
        Drawer.format_plot()
        plt.title(title)
        # plt.plot([0, 250], [0, 250], 'black')
        cate_num = len(true_lr_list)
        scatters = []
        for i_cate in range(cate_num):
            scatters.append(plt.scatter(true_lr_list[i_cate], pred_lr_list[i_cate], s=50, color='black',
                                        marker='s', alpha=0.5))
        Drawer.set_example_bar_ticks()

        plt.tight_layout(rect=[0, 0, 0.98, 0.98])
        plt.savefig('paper_figures/example result_2.png')

    @staticmethod
    def set_example_bar_ticks():
        ax = plt.gca()
        ax.set_xlim(0, 200)
        ax.set_xticks(range(0, 201, 50))
        ax.set_xticklabels(range(0, 201, 50), fontdict=FONT_DICT_SMALL)
        ax.set_xlabel('VALR: Laboratory Force Plate (BW/s)', labelpad=10, fontdict=FONT_DICT_SMALL)

        ax.set_ylim(0, 200)
        y_range = range(0, 201, 50)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylabel('VALR: CNN with Single Shank IMU (BW/s)', labelpad=10, fontdict=FONT_DICT_SMALL)


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

        info_path = '../2_loading_rate/result_conclusion/' + date + '/step_result/' + date + '_step_info.csv'
        info_df = pd.read_csv(info_path, index_col=False)
        self._step_result_df = pd.concat([self._step_result_df, info_df['step type']], axis=1)

    def get_param_mean_std_of_trial_mean(self, param_name, trial_list, sub_id_list=None):
        """
        This function takes the average accuracy of all the selected trials
        :param param_name: 'absolute mean error', 'pearson correlation', or 'RMSE'
        :param trial_list: e.g. ['all trials']
        :param sub_id_list:
        :return:
        """
        param_array = self.get_param_values(param_name, trial_list, sub_id_list)
        param_mean, param_std = np.mean(param_array), np.std(param_array)
        return param_mean, param_std

    def get_param_mean_std_of_all_steps(self, sub_id_list, select_value, select_col_name='trial id'):
        """
        This function pulls the step of all the selected trials and compute accuracy
        :param select_value:
        :param sub_id_list:
        :return:
        """
        pearson_coeffs, RMSEs, mean_errors, absolute_mean_errors = [], [], [], []
        for sub_id in sub_id_list:
            sub_df = self._step_result_df[self._step_result_df['subject id'].isin([sub_id])]
            trial_sub_df = sub_df[sub_df[select_col_name].isin(select_value)]
            pearson_coeff, RMSE, mean_error, absolute_mean_error = Evaluation.get_all_scores(
                trial_sub_df['true LR'], trial_sub_df['predicted LR'], precision=3)
            pearson_coeffs.append(pearson_coeff)
            RMSEs.append(RMSE)
            mean_errors.append(mean_error)
            absolute_mean_errors.append(absolute_mean_error)
        return np.mean(pearson_coeffs), np.std(pearson_coeffs), pearson_coeffs,\
               np.mean(absolute_mean_errors), np.std(absolute_mean_errors)

    def get_NRMSE_mean_std_of_all_steps(self, sub_id_list, select_value, select_col_name='trial id'):
        NRMSEs = []
        for sub_id in sub_id_list:
            sub_df = self._step_result_df[self._step_result_df['subject id'].isin([sub_id])]
            trial_sub_df = sub_df[sub_df[select_col_name].isin(select_value)]
            _, RMSE, _, _ = Evaluation.get_all_scores(
                trial_sub_df['true LR'], trial_sub_df['predicted LR'], precision=3)
            sub_max, sub_min = np.max(trial_sub_df['true LR']), np.min(trial_sub_df['true LR'])
            NRMSEs.append(RMSE/(sub_max - sub_min)*100)
        return np.mean(NRMSEs), np.std(NRMSEs)

    # def get_lr_values_of_gait(self, sub_id_list, select_value, select_col_name='trial id'):
    #     """Return the step lr of selected gait"""
    #
    #     for sub_id in sub_id_list:
    #         sub_df = self._step_result_df[self._step_result_df['subject id'].isin([sub_id])]
    #         trial_sub_df = sub_df[sub_df[select_col_name].isin(select_value)]
    #         pearson_coeff, RMSE, mean_error, absolute_mean_error = Evaluation.get_all_scores(
    #             trial_sub_df['true LR'], trial_sub_df['predicted LR'], precision=3)

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

    def get_one_trial_NRMSE_mean_std(self, sub_id_list=None, trial_name='All trials'):
        # step 0, get values
        param_df = self._result_df[self._result_df['parameter name'] == 'RMSE']
        # remove the overall mean row
        param_df = param_df[param_df['subject name'] != 'absolute mean']
        if sub_id_list is not None:
            sub_name_list = [SUB_NAMES[sub_id] for sub_id in sub_id_list]
            param_df = param_df[param_df['subject name'].isin(sub_name_list)]
        NRMSE_list = []
        for sub_name, RMSE in zip(param_df['subject name'], param_df[trial_name]):
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
            mean_NRMSE, _ = result_reader.get_one_trial_NRMSE_mean_std()
            if combo_best_mean[0] == -1 or mean_NRMSE < combo_best_mean[1]:
                combo_best_mean[0], combo_best_std[0] = result_reader.get_param_mean_std_of_trial_mean('pearson correlation',
                                                                                                 ['All trials'])
                combo_best_mean[1], combo_best_std[1] = result_reader.get_one_trial_NRMSE_mean_std()
                combo_best_mean[2], combo_best_std[2] = result_reader.get_param_mean_std_of_trial_mean('absolute mean error',
                                                                                                 ['All trials'])
                best_combo = combo
        return combo_best_mean, combo_best_std, best_combo
