import numpy as np
from const import SUB_NAMES
from Drawer import ResultReader
from sklearn.metrics import r2_score
from scipy import stats


# #creating OLS regression
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)


result_date = '1028'
reader = ResultReader(result_date, ['l_shank'])
result_df = reader._step_result_df

result_true = result_df['true LR'].values
result_pred = result_df['predicted LR'].values
_, _, r_value, _, _ = stats.linregress(result_true, result_pred)
print(r_value ** 2)

r2_sub = []
for i_sub in range(len(SUB_NAMES)):
    sub_result_df = result_df[result_df['subject id'] == i_sub]
    result_true = sub_result_df['true LR'].values
    result_pred = sub_result_df['predicted LR'].values
    _, _, r_value, _, _ = stats.linregress(result_true, result_pred)
    r2_sub.append(r_value ** 2)

print(r2_sub)
print(np.mean(r2_sub))


# import numpy as np
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# from scipy import stats
#
# #creating data
# x = np.array([0,1,2,3,4,5,6,7,8,9])
# y = np.array([0,2,3,5,8,13,21,34,55,89])
#
# #creating OLS regression
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# def linefitline(b):
#     return intercept + slope * b
# line1 = linefitline(x)
#
# #plot line
# plt.scatter(x,y)
# plt.plot(x,line1, c = 'g')
# line2 = np.full(10,[y.mean()])
# plt.scatter(x,y)
# plt.plot(x,line2, c = 'r')
#
# differences_line1 = linefitline(x)-y
# line1sum = 0
# for i in differences_line1:
#     line1sum = line1sum + (i*i)
#
# differences_line2 = line2 - y
# line2sum = 0
#
# for i in differences_line2:
#     line2sum = line2sum + (i*i)
#
# r2_all = 1 - line1sum / line2sum
# print(r2_all)
#
# r2 = r2_score(y, x)
# print('The rsquared value is: ' + str(r2))