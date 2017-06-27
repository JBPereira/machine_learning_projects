import random

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout, advanced_activations
from keras.layers.noise import GaussianNoise
from keras.models import Sequential, load_model
from sklearn.model_selection import KFold

from data_normalizer import DataNormalizer
from users_questionnaire.training_set import trainingset


class DeepLearner:
    def __init__(self, params=None, method=None, layers=None):

        self.layers = layers
        if params is not None:
            self.normalizer = DataNormalizer(params)
        if method is not None:
            self.method = method

    def build_model(self, nfeatures, activation='relu'):
        model = Sequential()

        layers = np.ones(2 * nfeatures) * 100

        depth = len(layers)

        for i in range(depth):

            if i is 0:
                # Input Layer of the Model
                # act = advanced_activations.PReLU(alpha_initializer='zeros')
                # act2 = advanced_activations.ThresholdedReLU(theta=-100)
                act = advanced_activations.ELU(alpha=0.8)
                # model.add(Dense(int(layers[i]), input_shape=(2 * nfeatures,), kernel_initializer='uniform'))

                model.add(Dense(int(layers[i]), input_shape=(2 * nfeatures,)))

                model.add(Dropout(.2))
                model.add(GaussianNoise(0.03))

                # model.add(act2)

                model.add(act)
                # model.add(Activation("tanh"))

            else:
                # Hidden Layers
                model.add(Dense(int(layers[i]), activation=activation))

                model.add(Dropout(.2))
                model.add(GaussianNoise(0.03))
                model.add(act)
                model.add(Activation("tanh"))
                # Output Layer
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

        self.model = model

    def train(self, training_data, training_class, activation='relu', batch_size=10, n_epoch=250,
              filename=False, pairwise=False,
              rank_reversed=True, double_ordering=True, verbose=2):

        if pairwise:
            nfeatures = len(training_data[0])/2
        else:
            nfeatures = len(training_data[0][0])

        # self.boost_model_config(epochs=n_epoch, batch_size=batch_size, n_estimators=n_estimators)

        if pairwise:
            pairwise_data, pairwise_class = training_data, training_class
        else:
            pairwise_data, pairwise_class = self.prepare_training_data(training_data, training_class,
                                                                       rank_reversed=rank_reversed)
        if double_ordering:
            pairwise_data = np.vstack([pairwise_data, pairwise_data[:, range(nfeatures, nfeatures*2) + range(0, nfeatures)]])
            pairwise_class = np.hstack([pairwise_class, 1-pairwise_class])
        self.build_model(nfeatures=nfeatures, activation=activation)
        self.model.fit(pairwise_data, pairwise_class, epochs=n_epoch, batch_size=batch_size, verbose=verbose)

        if filename:
            self.model.save('{} {}'.format(filename, i))

    def test_train(self, training_data, training_class, batch_size=10, n_epoch=200,
                   real_training_set=None):

        nfeatures = len(training_data[0][0])

        pairwise_data, pairwise_class = self.prepare_training_data(training_data, training_class)

        #### Getting original values to output results for users
        if real_training_set is not None:
            original_pairwise_data = self.case_pairwise_list(np.array(real_training_set[0]))
            for i in range(1, len(real_training_set)):
                pair_case = self.case_pairwise_list(np.array(real_training_set[i]))
                original_pairwise_data = np.concatenate((original_pairwise_data,
                                                         pair_case))

        n_splits = 2
        folder = KFold(n_splits=n_splits)
        data_splitter = folder.split(pairwise_data)
        cross_train_data = []
        cross_test_data = []
        cross_train_class = []
        cross_test_class = []
        original_test_data = []

        for train_data, test_data in data_splitter:
            cross_train_data += [pairwise_data[train_data]]
            cross_test_data += [pairwise_data[test_data]]
            cross_train_class += [pairwise_class[train_data]]
            cross_test_class += [pairwise_class[test_data]]
            #### Output results for users
            original_test_data += [original_pairwise_data[test_data]]

        n_cross_pieces = len(cross_train_data)

        accuracy_array = np.zeros(n_cross_pieces)

        for i in range(n_cross_pieces):

            ### Fit model to training data
            self.shuffle_weights()
            self.model.fit(cross_train_data[i], cross_train_class[i], batch_size=batch_size, epochs=n_epoch)

            ### Test accuracy on test data
            accuracy_array[i] = self.test_accuracy(cross_test_data[i], cross_test_class[i])

            ## Cycle to output real valued results
            for j in range(len(cross_test_data[i])):
                predicted = np.round(self.model.predict(np.reshape(cross_test_data[i][j], (1, 10))))
                cases = original_test_data[i][j]
                print "Case Pair number {} \n".format(j)
                for k in range(1, 3):
                    rank = [k - 1 == predicted]
                    rank[0] += 1
                    real_rank = [k - 1 == cross_test_class[i][j]]
                    real_rank[0] += 1
                    string = [k] + list(cases[(k - 1) * nfeatures:k * nfeatures]) + rank + real_rank
                    print "Line item {}: Budget = {}, DaysLeft = {}, PacingtoDate = {}, " \
                          "CPA= {}, PacingYesterday = {}, Rank = {} Real Rank = {}".format(*string)
                print "\n --------------------------------------- \n"

        avg_accuracy = sum(accuracy_array) / float(n_cross_pieces)
        print 'Average Accuracy' + str(avg_accuracy)
        return avg_accuracy
        # self.model.save('NN_trained_model.h5')

    def shuffle_weights(self, weights=None):
        """Randomly permute the weights in `model`, or the given `weights`.
        This is a fast approximation of re-initializing the weights of a model.
        Assumes weights are distributed independently of the dimensions of the weight tensors
          (i.e., the weights have the same distribution along each dimension).
        :param Model model: Modify the weights of the given model.
        :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
          If `None`, permute the model's current weights.
        """

        if weights is None:
            weights = [m.get_weights() for m in self.model]
        for index, m in enumerate(self.model):
            weights[index] = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
            m.set_weights(weights[index])

    def prepare_training_data(self, training_data, training_class=None, rank_reversed=True, return_index_list=False):

        """
        :param training_data: training data in format [[case1], [case2], ..., [caseN]] where case = [li1, li2, ..., lin]
        :param training_class: training class in format [case1_classes, ..., caseN_classes]
        :param rank_reversed: if the most important line item has the lowest score [1,2,3,4,..] then pass True
        :param return_index_list: the order of the li. within cases is shuffled (if passing in class) and everything is
        merged together afterwards. If you want to know the case for each pair in the list pass in True
        :return: List with pairs of line items. If return_index_list==True -> returns line items and case indexes as
        well in format [[li_number, li_number, case_number], ....]
        """

        ncases = len(training_data)
        if training_class is not None:
            if return_index_list:
                pairwise_train_data, pairwise_train_class, pairwise_ind_list = \
                    self.case_pairwise_list(np.array(training_data[0]), np.array(training_class[0]),
                                            rank_reversed=rank_reversed, return_index_list=return_index_list)
                pairwise_ind_list = np.insert(pairwise_ind_list, 0, 0, axis=1)
            else:
                pairwise_train_data, pairwise_train_class = \
                    self.case_pairwise_list(np.array(training_data[0]), np.array(training_class[0]),
                                            rank_reversed=rank_reversed)
        else:
            pairwise_train_data = \
                self.case_pairwise_list(np.array(training_data[0]),
                                        rank_reversed=rank_reversed)
        if ncases > 1:
            for i in range(1, ncases):
                if training_class is not None:
                    if return_index_list:
                        pairwise_case, pairwise_case_class, pairwise_case_ind_list = \
                            self.case_pairwise_list(np.array(training_data[i]), np.array(training_class[i]),
                                                    rank_reversed=rank_reversed, return_index_list=return_index_list)
                        pairwise_case_ind_list = np.insert(pairwise_case_ind_list, 0, i, axis=1)
                        pairwise_ind_list = np.concatenate((pairwise_ind_list, pairwise_case_ind_list), axis=0)
                    else:
                        pairwise_case, pairwise_case_class = \
                            self.case_pairwise_list(np.array(training_data[i]), np.array(training_class[i]),
                                                    rank_reversed=rank_reversed)

                    pairwise_train_class = np.concatenate((pairwise_train_class, pairwise_case_class), axis=0)

                else:
                    pairwise_case= \
                        self.case_pairwise_list(np.array(training_data[i]),
                                                rank_reversed=rank_reversed)
                pairwise_train_data = np.concatenate((pairwise_train_data, pairwise_case), axis=0)

        if training_class is not None:
            if return_index_list:
                return pairwise_train_data, pairwise_train_class, pairwise_ind_list
            else:
                return pairwise_train_data, pairwise_train_class
        else:
            return pairwise_train_data

    @staticmethod
    def case_pairwise_list(case, case_class=None, rank_reversed = True, return_index_list=False):
        """
        :param case: list of line items
        :param case_class: ranks of line items, to pass when training or testing. WARNING: in order to even out class representation,
        line items in group are shuffled. Take this into account if you want to rank.
        :return: List with all line items unique pairwise comparisons
        """

        nexamples = len(case)
        nfeatures = np.shape(case)[-1]
        len_pair_combinations = (nexamples * (nexamples - 1)) / 2  # sum [i=1:case_length] (i-1)
        case_index_list = np.zeros(shape=(len_pair_combinations, 2))
        pairwise_list = np.zeros(shape=(len_pair_combinations, 2 * nfeatures))

        if case_class is not None:
            # Shuffle cases to prevent class over-representation
            permutation = np.random.permutation(len(case))
            case = case[permutation]
            pairwise_class = np.zeros(len_pair_combinations, dtype=np.int8)
            case_class = case_class[permutation]

        for i in range(nexamples):
            relative_position = len_pair_combinations - ((nexamples - i) * (nexamples - i - 1)) / 2
            for j in range(i + 1, nexamples):
                pairwise_list[relative_position + j - i - 1] = np.concatenate((case[i], case[j]))
                if case_class is not None:
                    if case_class[i] < case_class[j]:
                        if rank_reversed:
                            pairwise_class[relative_position + j - i - 1] = 1
                    elif not rank_reversed:
                        pairwise_class[relative_position + j - i - 1] = 1
                    case_index_list[relative_position + j - i - 1] = [permutation[i], permutation[j]]
        if case_class is not None:
            if return_index_list:
                return pairwise_list, pairwise_class, case_index_list
            else:
                return pairwise_list, pairwise_class
        else:
            return pairwise_list

    def test_accuracy(self, test_data, test_class):

        """
        :param test_data: List of test data with example pairs sublist (ex.: [[line item1, line item2], ...]
        :param test_class: Class of pairwise priorities (ex.: [1,0,0,1,1...])
        :return: accuracy of model on the test data
        """

        correct_cases = 0
        for i in range(len(test_data)):
            predicted = np.round(self.model.predict(np.reshape(test_data[i], (1, 10))))
            if predicted == test_class[i]:
                correct_cases += 1

        accuracy = float(correct_cases) / len(test_class)
        print accuracy
        return accuracy

    def load_trained_model(self):
        self.model = load_model('NN_trained_model.h5')

    def rank_items(self, items, pacing_positions=False, CPA_to_target=False, normalize = True):

        """
        :param items: List of examples in case to be ranked, in tuples of example pairs (ex. [[[2,1,0.2], [0.1,2,2]],...]
        :return:
        """
        nfeatures = np.shape(items)[-1]
        if hasattr(self, 'method'):
            method = self.method
        else:
            method = 'min_max'
        if normalize:
            norm_items = DataNormalizer.normalize_data(items, method,
                                                   pacing_positions=pacing_positions,
                                                   CPA_target_pos=CPA_to_target)
            pairwise_list = self.case_pairwise_list(norm_items)
        else:
            pairwise_list = self.case_pairwise_list(items)
        nitems = len(items)
        n_pairs = len(pairwise_list)

        class_matrix = np.zeros((nitems, nitems))
        predicted_rank = np.zeros(nitems)
        for i in range(nitems):
            relative_position = n_pairs - ((nitems - i) * (nitems - i - 1)) / 2
            for j in range(i + 1, nitems):
                prediction = self.model.predict(np.reshape(pairwise_list[relative_position + j - i - 1], (1, 2*nfeatures)))
                class_matrix[i, j] = np.round(prediction)
                class_matrix[j, i] = 1 - class_matrix[i, j]

        for i in range(nitems):
            predicted_rank[i] = sum(class_matrix[i, :])

        return predicted_rank

    def rank_raw_items(self, data):

        data_matrix, valid_data, invalid_data = self.normalizer.process_li_data(data)
        data_matrix = np.array(data_matrix)

        predicted_rank = self.rank_items(data_matrix)
        rank = sorted(range(len(valid_data)), key=lambda k: predicted_rank[k], reverse=False)
        prio_valid_data = [valid_data[int(k)] for k in rank]

        return prio_valid_data, invalid_data

    def print_test_results(self, data):

        valid_data, invalid_data = self.rank_raw_items(data)

        for li in range(len(valid_data)):
            print 'Priority number {} : {}'.format(li, valid_data[li])


if __name__ == "__main__":

    NN = DeepLearner()

    params = ['network_budget', 'days_left', 'pacing_to_date', 'cpa_actual', 'pacing_yesterday', 'cpa_target']
    NN = DeepLearner(params=params, method='min-max')

    pacing_positions = [params.index('pacing_to_date'), params.index('pacing_yesterday') ]
    cpa_to_target_pos = params.index('cpa_actual')
    ts, training_class = NN.normalizer.process_training_set(trainingset, diff_to_target=False)

    norm_ts_hundred = NN.normalizer.normalize_training_data(ts, pacing_positions=pacing_positions, method='z-norm', CPA_target_pos=False)

    ############### Print Average Performance for various Training Sizes #######

    early_stopping = EarlyStopping(monitor='mse', patience=10)
    best_training_size = 0
    ensemble_number = 8
    best_avg_accuracy = 0
    lowest_ts = 5
    highest_ts = 23
    training_sizes = range(lowest_ts, highest_ts)
    performance_per_ts = np.zeros(np.shape(training_sizes))
    n_inner_cycles = 8
    for training_size in training_sizes:
        avg_accuracy = 0
        for cycle in range(n_inner_cycles):

            dummy_ts = np.array(norm_ts_hundred)
            dummy_class = np.array(training_class)
            test_NN = DeepLearner(params=params, method='z-norm')
            random_points = random.sample(range(len(dummy_ts)), training_size)
            train_data = dummy_ts[random_points]
            test_data = np.delete(dummy_ts, random_points, axis=0)
            train_class = dummy_class[random_points]
            test_class = np.delete(dummy_class, random_points, axis=0)

            test_NN.train(train_data, train_class, batch_size=10, n_epoch=10+training_size)
            pairwise_test, pairwise_test_class = test_NN.prepare_training_data(test_data, test_class)
            prediction = np.zeros(np.shape(pairwise_test_class))
            accuracy = 0
            for i in range(len(pairwise_test_class)):
                prediction[i] = np.round((test_NN.model.predict(np.reshape(pairwise_test[i], (1, 10)))))
                if prediction[i] == pairwise_test_class[i]:
                    accuracy +=1
            accuracy = (float(accuracy) / len(prediction)) *100
            print " \n\n ACCURACY: {}".format(accuracy)

            avg_accuracy += accuracy
        avg_accuracy = avg_accuracy / float(n_inner_cycles)
        print "AVERAGE ACCURACY: {}".format(avg_accuracy)
        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_training_size = training_size
        performance_per_ts[training_size-lowest_ts] = avg_accuracy

    np.save('avg_accuracy_per_ts', performance_per_ts)

    plt.plot(training_sizes, performance_per_ts)
    plt.ylabel('Average Accuracy Per Training Size')
    plt.xlabel('training_size')
    plt.show()






