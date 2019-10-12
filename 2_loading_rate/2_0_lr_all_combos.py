"""
This file does all the combos of five segments. Each combo was done using cross validation
"""
from ProcessorLR import ComboGenerator
from const import SUB_AND_SI_SR_TRIALS, RUNNING_TRIALS, SI_SR_TRIALS
import copy


# define train and test subjects
train = copy.deepcopy(SUB_AND_SI_SR_TRIALS)
# train = {'190521GongChangyang': SI_SR_TRIALS, '190522YangCan':  SI_SR_TRIALS}

ComboGenerator.all_combos(train, date='1010', c51=False, c52=True, c53=True)

