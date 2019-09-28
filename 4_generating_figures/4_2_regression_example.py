"""
Notes:
    1. In leave one out testing, yangcan's result was good.
"""
from plot_funcs import format_subplot
from ProcessorLR import ProcessorLR
from const import RUNNING_TRIALS, TRIAL_NAMES, FONT_DICT_SMALL, FONT_DICT, FONT_SIZE, SUB_AND_RUNNING_TRIALS
import copy
from Evaluation import Evaluation
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# train = {'190521GongChangyang': RUNNING_TRIALS[1:2]}
# test = {'190521GongChangyang': RUNNING_TRIALS[1:2]}
train = copy.deepcopy(SUB_AND_RUNNING_TRIALS)
del train['190423LiuSensen']
del train['190513YangYicheng']
del train['190513OuYangjue']
test = {'190423LiuSensen':  TRIAL_NAMES[1:2] + TRIAL_NAMES[3:5] + TRIAL_NAMES[6:7] + TRIAL_NAMES[8:14]}

my_LR_processor = ProcessorLR(train, test, 200, strike_off_from_IMU=2, split_train=False, do_output_norm=True)
predict_result_all = my_LR_processor.prepare_data()
my_LR_processor.define_cnn_model()

y_true, y_pred = my_LR_processor.to_generate_figure()


R2, RMSE, mean_error = Evaluation.get_all_scores(y_true, y_pred, precision=3)

plt.figure(figsize=(9, 6))
plt.plot(y_true, y_pred, 'b.')
plt.plot([0, 220], [0, 220], 'r--')
format_subplot()
RMSE_str = str(round(RMSE[0], 2))
mean_error_str = str(round(mean_error, 2))
pearson_coeff = str(round(pearsonr(y_true, y_pred)[0], 2))
plt.title('Correlation: ' + pearson_coeff, fontdict=FONT_DICT_SMALL)

plt.xlabel('true value (BW/s)', fontdict=FONT_DICT_SMALL)
plt.ylabel('predicted value (BW/s)', fontdict=FONT_DICT_SMALL)
ax = plt.gca()
ax.set_xlim(0, 220)
ax.set_xticks(range(0, 201, 50))
ax.set_xticklabels(range(0, 201, 50), fontdict=FONT_DICT_SMALL)

ax.set_ylim(0, 220)
ax.set_yticks(range(0, 201, 50))
ax.set_yticklabels(range(0, 201, 50), fontdict=FONT_DICT_SMALL)
plt.savefig('exported_figures/lr.png')
plt.show()












