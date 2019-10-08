import pandas as pd
import numpy as np


data = pd.read_csv('clip start end grid search result.csv')

result = data['correlation'].values[:-1]
result = result.reshape([10, 10])
end_summary = np.mean(result, axis=0)
start_summary = np.mean(result[:, 2:], axis=1)

x = 1


