import numpy as np
from scipy import signal
from sklearn.ensemble import IsolationForest

class DetectorUnit():

    def __init__(self, window_size=10, memory=None, memory_size=10, weights=None, fs=60, n_pseg=10):

        self.window_size = window_size
        self.fs = fs
        self.n_pseg = n_pseg
        self.stats_unit = IsolationForest(n_estimators=10, bootstrap=True)
        self.hold_state = False  # to allow frequency diff rebasing after alarm. float fft_diff is stored after alarm
        self.rebase_threshold = 1 # find a better method than hard thresholding
        self.memory_size = memory_size
        self.alarm_countdown = []
        self.weight_dev = 0.1
        if memory:
            self.memory = memory
        else:
            self.memory = np.array([0])[:,None]
        if weights:
            self.weights = weights
        else:
            self.weights = np.array([self.weight_dev])


    def compute_fft_high_diff(self, signal_in):

        size_signal = len(signal_in)
        lower_limit = max(size_signal-self.window_size, 0)

        f1, t1, Zxx1 = signal.stft(signal_in[lower_limit:size_signal-1],
                                   fs=self.fs, nperseg=self.n_pseg)
        f2, t2, Zxx2 = signal.stft(signal_in[lower_limit+1: size_signal],
                                   fs=self.fs, nperseg=self.n_pseg)

        return np.abs(Zxx2[-1,-1]) - np.abs(Zxx1[-1,-1])

    def update_stats_unit(self):

        self.stats_unit.fit(self.memory, sample_weight=self.weights)

    def forget_gate(self, new_fft_diff):

        # If memory is full penalize distant points and increase close points' weight
        # Keeps the most recent patterns and discards the rest. Accounts for changes in local stationarity

        if len(self.memory) >= self.memory_size:

            distance = self.memory - new_fft_diff

            max_ = np.max(distance)
            min_ = np.min(distance)
            distance = (distance - min_) / (max_ - min_)

            self.weights = np.subtract(self.weights, self.weight_dev * distance)
            self.weights[np.argmin(distance)] += 2 * self.weight_dev
            neg_values = self.weights < 0
            self.weights = np.delete(self.weights, neg_values)
            self.memory = np.delete(self.memory, neg_values)

        if len(self.weights) < self.memory_size:
            self.memory = np.vstack((self.memory, [new_fft_diff]))
            self.weights = np.concatenate((self.weights, [self.weight_dev]))
            if len(self.weights) > 2:
                self.stats_unit.fit(self.memory, sample_weight=self.weights)

    def detect_spike(self, signal_in):

        # Account for frequency shift of outlier at the end of the time window
        alarm_queue = len(self.alarm_countdown)
        if alarm_queue > 0:
            for countdown in range(alarm_queue):
                self.alarm_countdown[countdown] -= 1

        fft_diff = self.compute_fft_high_diff(signal_in)
        if -1 in self.alarm_countdown:
            fft_diff *= 0.6
            self.alarm_countdown.pop()

        # Detect the meaningful spikes
        if not self.hold_state:
            if len(self.memory) > 2:
                alarm = 1 if self.stats_unit.predict(fft_diff) == -1 else 0
            else:
                alarm = 0

        else:
            rebasing = np.abs(fft_diff - self.hold_state)
            if rebasing > np.abs(self.rebase_threshold * self.hold_state):
                alarm = 0
                self.hold_state = False
            else:
                alarm = 1

        # Refresh the memory of the unit
        if alarm:
            self.hold_state = fft_diff
            self.alarm_countdown.insert(0, self.n_pseg)
            self.forget_gate(fft_diff)
        else:
            self.forget_gate(fft_diff)

        return alarm

    def save_memory(self):
        pass
