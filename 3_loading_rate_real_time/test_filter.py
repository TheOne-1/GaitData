from scipy import zeros, signal, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_sbs():
    data_path = 'D:\Tian\Research\Projects\HuaweiProject\SharedDocs\Huawei\PhaseIData\ProcessedData\\' + \
                '190521GongChangyang\\100Hz\\nike SI 28.csv'
    data = pd.read_csv(data_path, index_col=False)
    acc_data = data['r_foot_acc_z'].values

    b = signal.firwin(100, 0.1)
    zi = signal.lfilter_zi(b, 1)
    acc_z = zeros(acc_data.size)
    for i, x in enumerate(acc_data):
        acc_z[i], zi = signal.lfilter(b, 1, [-x], zi=zi)

    b = signal.firwin(100, 0.9)
    zi = signal.lfilter_zi(b, 1)
    gyr_x = zeros(acc_data.size)
    for i, x in enumerate(acc_data):
        gyr_x[i], zi = signal.lfilter(b, 1, [-x], zi=zi)

    # gyr_data = data['r_foot_gyr_x'].values
    # b = signal.firwin(100, 0.9)
    # zi = signal.lfilter_zi(b, 1)
    # gyr_x = zeros(gyr_data.size)
    # for i, x in enumerate(gyr_data):
    #     gyr_x[i], zi = signal.lfilter(b, 1, [-x], zi=zi)
    #
    # b, a = signal.butter(4, 0.2, 'low')
    # fir_result = zeros(acc_data.size)
    # zi = signal.lfilter_zi(b, a)
    # for i, x in enumerate(acc_data):
    #     fir_result[i], zi = signal.lfilter(b, a, [x], zi=zi)
    #
    # b, a = signal.butter(4, 0.05, 'low')
    # iir_result = zeros(acc_data.size)
    # zi = signal.lfilter_zi(b, a)
    # for i, x in enumerate(acc_data):
    #     iir_result[i], zi = signal.lfilter(b, a, [x], zi=zi)

    return acc_data, acc_z, gyr_x


if __name__ == '__main__':
    data, fir_result, iir_result = filter_sbs()
    plt.plot(-data)
    plt.plot(fir_result)
    plt.plot(iir_result)
    plt.grid()
    plt.show()


