import numpy as np
from users_questionnaire.abigail import abigail_regular_data as abigail_data
from users_questionnaire.peer_data import peer_regular_data as peer_data
from users_questionnaire.rhein_data import rhein_regular_data as rhein_data
from users_questionnaire.sietse_data import sietse_regular_data as sietse_data

from NN_ranker import DeepLearner
from users_questionnaire.average_results_search import avg_regular_data as search_data
from users_questionnaire.rick_data import rick_regular_data as rick_data
from users_questionnaire.mark_data import mark_regular_data as mark_data
from users_questionnaire.training_set import trainingset as ts

params = ['network_budget', 'days_left', 'pacing_yesterday', 'pacing_to_date', 'cpa_actual', 'cpa_target']

pacing_positions = [params.index('pacing_yesterday'), params.index('pacing_to_date')]
cpa_target_pos = params.index('cpa_actual')
n_models = 1
NN = DeepLearner(params=params, method='z-norm', ensemble_number=n_models)

processed_ts, ts_class = NN.normalizer.process_training_set(ts, diff_to_target=False)

def process_individual_quesearch_data(data):
    class_ = []
    for i in range(len(data)):
        class__ = []
        for j in range(len(data[i])):
            class__.append(data[i][j][6])
            data[i][j][4] = data[i][j][4] / float(data[i][j][5])
            data[i][j] = data[i][j][0:5]
        class_.append(1 - np.array(class__))
    return class_, data

### Process Individual Questionnaire answers

# pbu
peer_class, peer_data = process_individual_quesearch_data(peer_data)
rick_class, rick_data = process_individual_quesearch_data(rick_data)
mark_class, mark_data = process_individual_quesearch_data(mark_data)

# search
search_class, search_data = process_individual_quesearch_data(search_data)
sietse_class, sietse_data = process_individual_quesearch_data(sietse_data)
rhein_class, rhein_data = process_individual_quesearch_data(rhein_data)
abigail_class, abigail_data = process_individual_quesearch_data(abigail_data)


w_cpa1 = np.genfromtxt('baas_search_data/BM - Search', delimiter=',')
w_cpa2 = np.genfromtxt('baas_search_data/BM-search2_data', delimiter=',')
w_cpa3 = np.genfromtxt('baas_search_data/BM-search3_data', delimiter=',')

combined_unlabeled = np.vstack((w_cpa1, w_cpa2, w_cpa3))
combined_unlabeled_cpa_ratio = np.copy(combined_unlabeled)
combined_unlabeled_cpa_ratio[:, 4] = combined_unlabeled_cpa_ratio[:, 4] / combined_unlabeled_cpa_ratio[:, 5]
combined_unlabeled_cpa_ratio = combined_unlabeled_cpa_ratio[:, 0:5]


norm_wcpa1 = NN.normalizer.normalize_data(w_cpa1, method='z-norm', pacing_positions=pacing_positions,
                                                      CPA_target_pos=cpa_target_pos)

norm_combined_unlabeled = NN.normalizer.normalize_data(combined_unlabeled_cpa_ratio, method='z-norm', pacing_positions=pacing_positions,
                                                      CPA_target_pos=cpa_target_pos)


search_names = ['sietse', 'rhein', 'abigail', 'search']
pbu_names = ['mark', 'rick', 'peer']

for name in search_names:

    NN = DeepLearner(params=params, method='z-norm', ensemble_number=n_models)

    combined_data = eval(name+'_data') + processed_ts
    combined_data = np.array(combined_data)
    combined_class = eval(name+'_class') + ts_class
    combined_class = np.array(combined_class)

    norm_combined = NN.normalizer.normalize_training_data(combined_data, method='z-norm', pacing_positions=pacing_positions,
                                                          CPA_target_pos=cpa_target_pos) ## Normalizes ddata

    pair_combined, pair_combined_class = NN.prepare_training_data(norm_combined, combined_class)

    NN.train(pair_combined, pair_combined_class, rank_reversed=True, n_epoch=400, pairwise=True, double_ordering=True, verbose=2)



    rank_order = NN.rank_items(norm_combined_unlabeled, pacing_positions=pacing_positions,
                                                          CPA_to_target=cpa_target_pos, normalize=False)
    rank = sorted(range(len(combined_unlabeled)), key=lambda k: rank_order[k], reverse=False)
    rank = np.array(rank)
    prio_valid_data = combined_unlabeled[rank.astype(int)]

    params = ['network_budget', 'days_left', 'pacing_yesterday', 'pacing_to_date', 'CPA_to_target_ratio']
    print "{} Prioritization \n".format(name)
    for i in rank_order:
        # print ["{} : {}".format(a_, b_) for a_, b_ in zip(params, i)]
        print i
combined_unlabeled[:,2] /= 100
combined_unlabeled[:,3] /= 100
for i in combined_unlabeled:
    array = ["%.2f" % j for j in i]
    print array

############ Compute the correlation matrix between TS points and Unlabeled points ############

# dis_matrix, pair_ts_index = compute_inter_pair_distance_matrix(combined_data, combined_class,
#                                                                w_cpa1, params=params, pacing_pos=pacing_positions, cpa_target_pos=cpa_target_pos,
#                                                                norm_method='z-norm')
# first_ = 16
# second_ = 19
# original_pairs, class_cases = \
#     closest_pair(dis_matrix,
#                  combined_data,
#                  combined_class,
#                  (first_, second_), pair_ts_index)
#
# print ["{} : {}".format(a_, b_) for a_, b_ in zip(params, w_cpa1[first_])]
# print ["{} : {}".format(a_, b_) for a_, b_ in zip(params, w_cpa1[second_])]
# print "Closest Pairs"
# for i in range(len(original_pairs)):
#     print "\n"
#     print ["{} : {}".format(a_, b_) for a_, b_ in zip(params, original_pairs[i][0])]
#     print ["{} : {}".format(a_, b_) for a_, b_ in zip(params, original_pairs[i][1])]
#     print class_cases[i]

###################################### Test Average Accuracy ######################

# n_cycles = 15
# avg_accuracy = 0
# std_ = []
# combined_data = search_data + processed_ts
# combined_data = np.array(combined_data)
# combined_class = search_class + ts_class
# combined_class = np.array(combined_class)
# for cycle in range(n_cycles):
#     accuracy = 0
#     NN = DeepLearner(params=params, method='z-norm', ensemble_number=n_models)
#     norm_combined = NN.normalizer.normalize_training_data(combined_data, method='z-norm', pacing_positions=pacing_positions,
#                                                           CPA_target_pos=cpa_target_pos)
#     norm_combined = np.array(norm_combined)
#
#     random_pos = random.sample(range(len(norm_combined)), int(0.8 * len(norm_combined)))
#     train_set = norm_combined[random_pos]
#     # train_set = train_set[:, [0,1,2,3,5,6,7,8]]
#     train_class = combined_class[random_pos]
#     pair_combined_train, pair_class_train = NN.prepare_training_data(train_set, train_class)
#     # semi_supervised_learner = LabelSpreading(kernel='knn', n_neighbors=2)
#     # semi_supervised_learner.fit(train_set, train_class)
#     # unlabeled_predicted_label = semi_supervised_learner.predict(pair_unlabeled)
#     # unl_plus_train = np.vstack((train_set, pair_unlabeled))
#     # unl_plus_train_class = np.append(train_class, unlabeled_predicted_label)
#     # NN.train(pair_unlabeled, unlabeled_predicted_label, rank_reversed=True, n_epoch=3, double_ordering=True, pairwise=True)
#     NN.train(pair_combined_train, pair_class_train, rank_reversed=True, n_epoch=500, pairwise=True, double_ordering=True)
#     test_set = np.delete(norm_combined, random_pos, axis=0)
#     test_class = np.delete(combined_class, random_pos, axis=0)
#     pair_combined_test, pair_class_test = NN.prepare_training_data(test_set, test_class)
#     for item_index, item in enumerate(pair_combined_test):
#
#         prediction = np.round((NN.model.predict(np.reshape(item, (1, 10)))))
#
#         if pair_class_test[item_index] == prediction:
#             accuracy += 1
#     accuracy /= float(len(pair_class_test))
#     avg_accuracy += accuracy
#     std_.extend([accuracy])
#     print "ACCURACY Double Pairwise Order: {}".format(accuracy)
# avg_accuracy /= float(n_cycles)
# std_ = np.std(std_)
# print "\n\n\n AVG ACCURACY Gaussian(0.03) all layers after dropout no tanh at input doubletraining {}\n\n\n".format(avg_accuracy)
# print "\n STD {} \n".format(std_)







