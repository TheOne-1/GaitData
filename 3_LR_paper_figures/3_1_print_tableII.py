
import numpy as np
from Drawer import Drawer, ResultReader, ComboResultReader
from ComboGenerator import ComboGenerator
import copy
from const import SI_SR_TRIALS


segments = ['l_shank']
result_date = '1013'
precision = 3

the_reader = ResultReader(result_date, segments)
trial_shoe_0_speed_24 = list(SI_SR_TRIALS[:2])
shoe_0_speed_24 = the_reader.get_param_mean_std('absolute mean error', trial_shoe_0_speed_24)
x = 1


