import pandas as pd
import numpy as np


def combinations_by_subset(seq, r):
    if r:
        for i in range(r - 1, len(seq)):
            for cl in combinations_by_subset(seq[:i], r - 1):
                yield cl + (seq[i],)
    else:
        yield tuple()

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


