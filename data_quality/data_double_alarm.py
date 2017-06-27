import numpy as np
from scipy import signal
from sklearn.cluster import KMeans

class DetectorUnit():

    def __init__(self, window_size=10, memory=None, memory_diff=None,
                 memory_size=20, memory_diff_size=10, weights=None):

        self.window_size = window_size
        self.stats_unit = KMeans(n_clusters=memory_size)
        self.stats_unit_diff = KMeans(n_clusters=memory_size)
        self.memory_size = memory_size
        self.memory_diff_size = memory_diff_size
        self.double_prior = 0.03
        if memory:
            self.memory = memory
        else:
            self.memory = np.array([0])[:, None]
            self.memory_std = np.array([0])[:, None]
        if memory_diff:
            self.memory_diff = memory_diff

        else:
            self.memory_diff = np.array([0])[:, None]
            self.memory_diff_std = np.array([0])[:, None] # Do not forget to do something about memory as input (cluster memory into fixed size memory unit and keep the std)

        if weights:
            self.weights = weights
        else:
            self.weights = np.array([self.weight_dev])
        self.weights_diff = np.array([self.weight_dev])


    def compute_first_order_diff(self, signal_in):

        first_order_diff = signal_in[-1] - signal_in[-2]

        return first_order_diff

    def update_stats_unit(self, new_value, memory):

        if memory == 'value':
            mem = self.memory
            unit = self.stats_unit
            w = self.weights
        else:
            mem = self.memory_diff
            unit = self.stats_unit_diff
            w = self.weights_diff

        # Cluster points in size_memory groups

        unit.fit([mem, new_value])
        labels = unit.labels_

        vals, inverse, count = np.unique(labels, return_inverse=True,
                                      return_counts=True)

        idx_vals_repeated = np.where(count > 1)[0]
        vals_repeated = vals[idx_vals_repeated]

        rows, cols = np.where(inverse == idx_vals_repeated[:, 0])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:])

        mem[res[0]] = (1/(w[res[0]]+1)) * (w[res[0]] * mem[res[0]] +
                                     w[res[1]])  # update value to the new mean

        w[res[0]] += w[res[1]]
        w[res[1]] = 1

        mem[res[1]] = new_value

        return mem, w



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

        if signal_in[-1] < 2:
            return 1

        else:

            diff = self.compute_first_order_diff(signal_in)

            alarm = self.posterior_probability(diff, signal_in[-1])
            # dont forget to push the new values to memory

            if alarm > 0.5:
                return 1
            else:
                self.add_new_value(signal_in[-1], diff)

                return 0

    def add_new_value(self, value, diff):

        diff_dist_to_mem = diff - self.memory_diff
        dist_to_mem = value - self.memory

        # If the new value is in the range of the already seen values, update memory, else:

        if min(diff_dist_to_mem) > np.mean(self.memory_diff) + np.std(self.memory_diff):
            self.memory_diff, self.weights_diff = self.update_stats_unit(diff, 'diff')
        else:
            min_mem_slot = np.argmin(diff_dist_to_mem)
            old_mean_part = self.memory_diff[min_mem_slot] * self.weights[min_mem_slot]
            self.memory_diff[min_mem_slot] = (old_mean_part * self.weights[min_mem_slot]
                                              + value) / (self.weights[min_mem_slot] + 1)
            self.weights_diff[min_mem_slot] += 1
        
        if min(dist_to_mem) > np.mean(self.memory) + np.std(self.memory):
            self.memory, self.weights = self.update_stats_unit(value, 'value')
        else:
            min_mem_slot = np.argmin(dist_to_mem)
            old_mean_part = self.memory[min_mem_slot] * self.weights[min_mem_slot]
            self.memory[min_mem_slot] = (old_mean_part * self.weights[min_mem_slot] + value)\
                                        / (self.weights[min_mem_slot] + 1)
            self.weights[min_mem_slot] += 1
        
        

    def posterior_probability(self, new_value, diff):

        possible_true_values = ((new_value - diff)
                                - self.memory_diff) - (new_value / 2.0)

        threshold = np.mean(possible_true_values) - (np.std(possible_true_values) / 2.0)  # Re-Define with standard deviation update

        if min(possible_true_values) < threshold:
            estimated_truevalue_index = np.argmin(possible_true_values)
            prob_possible_value = self.weights_diff[estimated_truevalue_index] / sum(self.weights_diff)
        else:
            prob_possible_value = 0.01  # bias

        half_values = (new_value / 2.0) - self.memory
        value_threshold = np.mean(half_values) - (np.std(half_values) / 2.0)
        if min(half_values) < value_threshold:
            estimated_half_value_index = np.argmin(half_values)
            prob_half_value = self.weights[estimated_half_value_index] / sum(self.weights)
        else:
            prob_half_value = 0.01  # bias

        evidence_values = new_value - self.memory
        value_threshold = np.mean(evidence_values) - (np.std(evidence_values) / 2.0)
        if min(evidence_values) < value_threshold:
            evidence_values_index = np.argmin(evidence_values)
            evidence = self.weights[evidence_values_index] / sum(self.weights)
        else:
            evidence = 1 / float(sum(self.weights) + 1)

        likelihood = prob_half_value + prob_possible_value

        posterior = (likelihood * self.double_prior) / evidence

        return posterior

    def calculate_new_std(self, old_std, new_value):

        pass


    def save_memory(self):
        pass