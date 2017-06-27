import numpy as np
from scipy.spatial import distance

def jsonfy_questionnaire_data(data):

    params = ['network_budget', 'days_left', 'pacing_yesterday', 'pacing_to_date', 'cpa_actual', 'cpa_target']

    json_data = {}

    for example_index in range(len(data)):
        json_data['Example {}'.format(example_index)] = []
        for li in range(len(data[example_index])):

            li_dict = dict()
            for par in range(len(params)+1):
                if par == len(params):
                    li_dict['Score'] = data[example_index][li][par]
                else:
                    li_dict[params[par]] = data[example_index][li][par]
            json_data['Example {}'.format(example_index)].append(li_dict)
    return json_data

def separate_spend_data(data):
    """
    :param data: From Questionnaire data, separate spend from regular data
    :return: Regular data, Spend data, ready to jsonfy
    """
    spend_data = []
    regular_data = []
    for case_index in range(len(data)):
        regular_buffer = []
        spend_buffer = []
        for example_index in range(len(data[case_index])):
            if min(data[case_index][example_index]) is None:
                spend_buffer.append(data[case_index][example_index])
            else:
                regular_buffer.append(data[case_index][example_index])
        if len(spend_buffer) > 1:
            spend_data.append(spend_buffer)
        elif len(regular_buffer) > 0 :
            regular_data.append(regular_buffer)
    return regular_data, spend_data

def duplicate_array_concatenation(pairwise_data):
    """
    Since the pairwise concatenation of data applies random ordering this insures all possibilities are covered
    :param pairwise_data: Pairwise data array
    :return: Pairwise data concatenated with duplicated inverse order data
    """

    du_pair_data = np.concatenate(pairwise_data, np.zeros(np.shape(pairwise_data)))

    n_arrays = len(pairwise_data)
    n_features = np.shape(pairwise_data)[1]

    for i in range(len(pairwise_data)):
        du_pair_data[n_arrays+i, :] = np.concatenate(pairwise_data[i, n_features:2*n_features],
                                                    pairwise_data[i, 0:n_features])
    return du_pair_data

def separate_pairwise_data(pairwise_data):
    """
    From Pairwise Data separate the compared arrays into two matrices
    :param pairwise_data: Pairwise matrix
    :return: Two matrices with the compared arrays in the pairwise data
    """

    n_features = np.shape(pairwise_data)[1]

    return pairwise_data[:, 0:n_features/2], pairwise_data[:, n_features/2:n_features]

def pairwise_inter_quartile_range(pairwise_data):
    """
    From the pairwise data, calculate how much of the feature space is covered. This is done by
    calculating the pair distances and returning their interquartile range
    :param pairwise_data:
    :return: inter_quartile_range of array pair distances
    """

    pair_one_data, pair_two_data = separate_pairwise_data(pairwise_data)
    pair_distance = distance.cdist(pair_one_data, pair_two_data)
    q75, q25 = np.percentile(pair_distance, [75, 25])
    iqr = q75 - q25

    return iqr



def assign_to_foreign_nn(data, centers, pairwise =False):
    """
    :param data: data to be assigned to the nearest neighbours centers
    :param centers: centers to which the points in the data will be assigned
    :param pairwise: if the data is made of concatenated array pairs, pass True to take it into account
    :return: label_array: Array with which center belongs each point in the data
             inter_iqr: the interquartile range for the minimum inter distances
             low_percentile
             high_percentile

    """

    n_samples_data, n_features_data = np.shape(data)
    n_samples_centers, n_features_centers = np.shape(centers)
    distance_matrix = np.zeros((n_samples_data, n_samples_centers))

    for i in range(n_samples_data):
        for j in range(n_samples_centers):
            if pairwise:
                data_inverted = np.concatenate((data[i, n_features_data/2:], data[i, 0: n_features_data/2]))
                center_inverted = np.concatenate((centers[j, n_features_centers/2:], centers[j, 0: n_features_centers/2]))
                distance_matrix[i, j] = min(distance.euclidean(data[i], centers[j]),
                                            distance.euclidean(data_inverted, centers[j]),
                                            distance.euclidean(data[i], center_inverted),
                                            distance.euclidean(data_inverted, center_inverted))
            else:
                distance_matrix[i, j] = distance.euclidean(data[i], centers[j])

    label_array = np.zeros((n_samples_data, 1))
    min_distances = np.zeros((n_samples_data, 1))

    for i in range(n_samples_data):
        label_array[i] = np.argmin(distance_matrix[i, :])
        min_distances[i] = np.min(distance_matrix[i, :])
    q75, q25 = np.percentile(min_distances, [75, 25])
    iqr = q75 - q25

    return label_array, iqr, q25, q75, min_distances