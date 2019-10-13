"""
This file does all the combos of five segments. Each combo was done using cross validation
"""
from ComboGenerator import ComboGenerator
from const import SUB_AND_SI_SR_TRIALS
import copy


# define train and test subjects
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
# train = {'190521GongChangyang': SI_SR_TRIALS, '190522QinZhun':  SI_SR_TRIALS}

ComboGenerator.all_combos(train, date='1013', c51=False, do_pta=True)

