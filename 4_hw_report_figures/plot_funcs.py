
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from const import LINE_WIDTH, FONT_DICT, FONT_DICT_SMALL


def format_subplot():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=LINE_WIDTH)
    ax.yaxis.set_tick_params(width=LINE_WIDTH)
    ax.spines['left'].set_linewidth(LINE_WIDTH)
    ax.spines['bottom'].set_linewidth(LINE_WIDTH)
