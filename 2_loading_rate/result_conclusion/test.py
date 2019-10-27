import pandas as pd
import numpy as np

tt = (sm-m)/np.sqrt(sv/float(n))  # t-statistic for mean
pval = stats.t.sf(np.abs(tt), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
print 't-statistic = %6.3f pvalue = %6.4f' % (tt, pval)
t-statistic =  0.391 pvalue = 0.6955

# segments = ['trunk', 'pelvis', 'l_thigh', 'l_shank', 'l_foot']
segments = 'ABCD'
x = combinations_by_subset(segments, 2)
for t in x:
    print(t)

data = pd.read_csv('predict_result_conclusion_4.csv')

result = data['correlation'].values[:-1]
result = result.reshape([10, 5])
end_summary = np.mean(result, axis=0)
start_summary = np.mean(result[:, 2:], axis=1)

x = 1


