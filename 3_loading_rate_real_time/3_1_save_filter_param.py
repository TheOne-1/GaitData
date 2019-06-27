"""
Export FIR filter parameter for C++ code
"""
import numpy as np
from scipy import signal
from const import TRIAL_START_BUFFER, FILTER_WIN_LEN
import json


def write_text_file(path, text):
    """Write a string to a file"""
    with open(path, "w") as text_file:
        print(text, file=text_file)


cut_off_fre = 10
sampling_fre = 200
filter_win_len = 100
param_file = 'filter_param.json'
wn = cut_off_fre / sampling_fre
b = signal.firwin(filter_win_len, wn)
a = 1

filter_delay = int(FILTER_WIN_LEN / 2)

filter_param = {'wn': wn, 'b': b.tolist(), 'a': a, 'filter_win_len': filter_win_len,
                'filter_delay': filter_delay, 'strike_delay': 8, 'off_delay': 6, 'start_buffer': TRIAL_START_BUFFER}
with open(param_file, 'w') as param_file:
    print(json.dumps(filter_param, sort_keys=True, indent=4, separators=(',', ': ')), file=param_file)



























