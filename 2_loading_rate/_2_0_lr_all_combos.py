"""
This file does all the combos of five segments. Each combo was done using cross validation
"""
from ComboGenerator import ComboGenerator
from const import SUB_AND_SI_SR_TRIALS, SI_SR_TRIALS
import copy


# define train and test subjects
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
# train = {'190521GongChangyang': SI_SR_TRIALS, '190523ZengJia':  SI_SR_TRIALS}

ComboGenerator.all_combos(train, date='1009', from_imu=2, c51=True)

