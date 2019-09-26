from const import TRIAL_NAMES, SUB_NAMES
import numpy as np
import pandas as pd


class AllSubDataStruct:
    """
    This is a simple data structure for pop or insert input, output, status all together
    """

    def __init__(self):
        self.__input_all_list, self.__output_all_list = [], []
        self.__sub_id_list, self.__trial_id_list = [], []
        self.__data_len = 0

    def __len__(self):
        return self.__data_len

    def append(self, input_list, output_list, subject_name, trial_name):
        if len(input_list) != len(output_list):
            raise ValueError('The length of input and output should be the same')
        self.__data_len += len(input_list)
        self.__input_all_list.extend(input_list)
        self.__output_all_list.extend(output_list)
        subject_id = SUB_NAMES.index(subject_name)
        trial_id = TRIAL_NAMES.index(trial_name)
        for i_item in range(len(input_list)):
            self.__sub_id_list.append(subject_id)
            self.__trial_id_list.append(trial_id)

    def pop(self, index):
        self.__input_all_list.pop(index)
        self.__output_all_list.pop(index)
        self.__sub_id_list.pop(index)
        self.__trial_id_list.pop(index)
        self.__data_len -= 1

    def get_input_output_list(self):
        return self.__input_all_list, self.__output_all_list

    def get_sub_id_list(self):
        return self.__sub_id_list

    def get_trial_id_list(self):
        return self.__trial_id_list
