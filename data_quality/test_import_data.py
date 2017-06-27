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
    freq_diff.append(np.abs(Zxx2[-1, -1]) - np.abs(Zxx1[-1, -1]))

plt.figure()
# plt.xticks(date_, date_array)
ax1 = plt.subplot(211)
ax1.plot(np.array(x_outliers), freq_diff, '--r')
ax1.plot(np.array(x_outliers), sorted_size[window_size: len(sorted_size)])


## Test the Data Quality alarm

detect_unit = DetectorUnit(window_size=5, memory_size=30, n_pseg=4)
outliers_ = []

for t in range(window_size, len(sorted_size)):
    alarm = detect_unit.detect_spike(sorted_size[0:t+1])
    outliers_.append(alarm)

ax2 = plt.subplot(212)

norm_size = (sorted_size-min(sorted_size))/float(max(sorted_size) - min(sorted_size))

ax2.plot(x_outliers, norm_size[x_outliers], label='Nrows in file Over Time')

ax2.plot(np.array(range(window_size, len(sorted_size))), outliers_, 'r', label='Alarm On (1) / Off (0)')
# ax2.xlabel('Days')
plt.legend(loc='lower left', bbox_to_anchor=(0.5, 1))
plt.show()

## Test variation over time  of the data



## Test Normality of the data

def EvalCdf(sample, x):
    count = 0.0
    for value in sample:
        if value <= x:
            count += 1
    prob = count / np.shape(sample)[0]
    return prob

normal = np.random.normal(np.mean(sorted_size), np.std(sorted_size), np.shape(sorted_size)[0])
bin_step = (max(sorted_size) - min(sorted_size))/(float(np.shape(sorted_size)[0]*0.1))
bins = np.arange(min(sorted_size) + bin_step, max(sorted_size)+bin_step, bin_step)
cdf = np.zeros(np.shape(bins))
cdf_normal = np.zeros(np.shape(bins))
for bin in range(len(bins)):
    cdf[bin] = EvalCdf(sorted_size, bins[bin])
    cdf_normal[bin] = EvalCdf(normal, bins[bin])

plt.figure()
plt.plot(bins, cdf)
plt.plot(bins, cdf_normal)

## Plot signal descriptive statistics over time

time_steps = np.arange(0, len(sorted_size), 10)
variation_over_time = np.zeros(np.shape(time_steps)[0])
variation_since_start = np.zeros(np.shape(time_steps)[0])
variation_over_time[0] = np.std(sorted_size[0:time_steps[1]])
variation_since_start[0] = variation_over_time[0]
mean_over_time = np.zeros(np.shape(time_steps)[0])
mean_since_start = np.zeros(np.shape(time_steps)[0])
mean_over_time[0] = np.std(sorted_size[0:time_steps[1]])
mean_since_start[0] = mean_over_time[0]
for bin in range(1, np.shape(time_steps)[0]):
    variation_over_time[bin] = np.std(sorted_size[time_steps[bin-1]:time_steps[bin]])
    variation_since_start[bin] = np.std(sorted_size[0 : time_steps[bin]])
    mean_over_time[bin] = np.mean(sorted_size[time_steps[bin-1]:time_steps[bin]])
    mean_since_start[bin] = np.mean(sorted_size[0 : time_steps[bin]])

diffs = np.zeros(len(sorted_size)-1)
log_diffs = np.zeros(len(sorted_size)-1)

for t in range(len(diffs)):
    diffs[t] = sorted_size[t+1] - sorted_size[t]
    log_diffs[t] = np.log(diffs[t])
plt.close('all')
plt.figure()
ax1 = plt.subplot(311)
ax1.plot(time_steps , variation_over_time)
ax1.plot(time_steps, [np.std(sorted_size)] * len(time_steps), '--r')
ax1.plot(time_steps, variation_since_start, 'og')
ax2 = plt.subplot(312)
ax2.plot(time_steps , mean_over_time)
ax2.plot(time_steps, [np.mean(sorted_size)] * len(time_steps), '--r')
ax2.plot(time_steps, mean_since_start, 'og')
ax3 = plt.subplot(313)
ax3.plot(diffs)
ax3.plot(log_diffs, '--r')