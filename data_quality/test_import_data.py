from process_data_over_time import get_size, sort_by_time
from data_alarm import DetectorUnit
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

## Getting and processing test data

# date_array, size_array = get_size('/home/joao/Desktop/adwords/extract/2017/', import_account='3692849778')
date_array = np.load('test_date_array.npy')
size_array = np.load('size_array.npy')
# np.save('test_date_array', date_array)
# np.save('size_array', size_array)

date_, sorted_size = sort_by_time(date_array, size_array)


plt.close('all')

## Plot Frequency shift over time

window_size = 10
x_outliers = np.arange(window_size, len(sorted_size))

freq_diff = []

for t in range(window_size, len(sorted_size)):

    f1, t1, Zxx1 = signal.stft(sorted_size[t-window_size:t], fs=120, nperseg=5)
    f2, t2, Zxx2 = signal.stft(sorted_size[t-window_size+1:t+1], fs=120, nperseg=5)
    freq_diff.append(np.abs(Zxx2[-1,-1]) - np.abs(Zxx1[-1,-1]))

plt.figure()
plt.xticks(date_, date_array)
plt.plot(np.array(x_outliers), freq_diff, '--r')
plt.plot(np.array(x_outliers), sorted_size[window_size: len(sorted_size)])
plt.show()

## Test the Data Quality alarm

detect_unit = DetectorUnit(window_size=5, memory_size=30, n_pseg=4)
outliers_ = []

for t in range(window_size, len(sorted_size)):
    alarm = detect_unit.detect_spike(sorted_size[0:t+1])
    outliers_.append(alarm)

fig = plt.figure()

norm_size = (sorted_size-min(sorted_size))/float(max(sorted_size) - min(sorted_size))

plt.plot(x_outliers, norm_size[x_outliers], label='Nrows in file Over Time')

plt.plot(np.array(range(window_size, len(sorted_size))), outliers_, 'r', label='Alarm On (1) / Off (0)')
plt.xlabel('Days')
plt.legend(loc='lower left', bbox_to_anchor=(0.5, 1))
plt.show()
