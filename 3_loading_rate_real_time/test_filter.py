from scipy import zeros, signal, random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_sbs():
    data_path = 'D:\Tian\Research\Projects\HuaweiProject\SharedDocs\Huawei\PhaseIData\ProcessedData\\' + \
                '190521GongChangyang\\100Hz\\nike SI 28.csv'
    data = pd.read_csv(data_path, index_col=False)
    acc_data = data['r_foot_acc_z'].values
    acc_z = zeros(acc_data.size)

    b = signal.firwin(100, 0.1)
    zi = signal.lfilter_zi(b, 1)
    for i, x in enumerate(acc_data):
        acc_z[i], zi = signal.lfilter(b, 1, [-x], zi=zi)

    b = signal.firwin(100, 0.1)
    zi = signal.lfilter_zi(b, 1)
    gyr_x = signal.lfilter(b, 1, -acc_data)


    # b = signal.firwin(100, 0.99)
    # zi = signal.lfilter_zi(b, 1)
    # gyr_x = zeros(acc_data.size)
    # for i, x in enumerate(acc_data):
    #     gyr_x[i], zi = signal.lfilter(b, 1, [-x], zi=zi)

    return acc_data, acc_z, gyr_x


if __name__ == '__main__':
    data, fir_result, iir_result = filter_sbs()
    plt.plot(-data)
    plt.plot(fir_result)
    plt.plot(iir_result)
    plt.grid()
    plt.show()


# from scipy import signal
# import matplotlib.pyplot as plt
# import numpy as np
#
# t = np.linspace(-1, 1, 201)
# x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) + 0.1*np.sin(2*np.pi*1.25*t + 1) + 0.18*np.cos(2*np.pi*3.85*t))
# xn = x + np.random.randn(len(t)) * 0.08
# # Create an order 3 lowpass butterworth filter:
#
#
# b, a = signal.butter(3, 0.05)
# zi = signal.lfilter_zi(b, a)
# z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
# z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
# y = signal.filtfilt(b, a, xn)
# # Plot the original signal and the various filtered versions:
#
# plt.figure()
# plt.plot(t, xn, 'b', alpha=0.75)
# plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
# plt.grid(True)
# plt.show()