import numpy as np

class DataNormalizer:
    def __init__(self, params, normalizing_factor=0):
        self.normalizing_factor = normalizing_factor
        self.params = params

    def process_training_set(self, json_data, diff_to_target=False):

        """
        :param json_data: Json data with a set of cases with multiple line items and their priority
        :param diff_to_target:
        :return: Processed data and an array of their classes
        """

        ready_training_set = []
        classes = []
        cpa_index = self.params.index('cpa_actual')
        if 'cpa_target' in self.params:
            line_length = len(self.params) - 1
            target_cpa_check = True
        else:
            line_length = len(self.params)
            target_cpa_check = False
        for case in json_data:
            set = []
            question_classes = []
            for example in json_data[case]:

                new_line = [0] * line_length
                for par, value in example.iteritems():
                    if par == 'Score':
                        question_classes += [value]
                    else:
                        if par in self.params:
                            par_index = self.params.index(par)
                            if target_cpa_check:
                                if par == 'cpa_target':
                                    if value is None:
                                        line_item_tcpa = value
                                    else:
                                        line_item_tcpa = float(value)
                                else:
                                    new_line[par_index] = float(value)
                            else:
                                new_line[par_index] = float(value)
                if target_cpa_check:
                    if line_item_tcpa is None:
                        adjusted_cpa_to_target_value = -np.inf
                    else:
                        adjusted_cpa_to_target_value = new_line[cpa_index] - line_item_tcpa
                    new_line[cpa_index] = adjusted_cpa_to_target_value
                set.append(new_line)
            ready_training_set.append(set)
            classes.append(question_classes)
        return ready_training_set, classes

    def process_li_data(self, data, missing_CPA=False):

        """
        :param data: Already formatted data
        :param missing_CPA:
        :return: Processed data ready to normalize or use in the Classifier
        """

        validata = []
        invalidata = []
        matrix_data = []
        nfeatures = len(self.params)

        if 'cpa_target' in self.params:
            target_check = True
        else:
            target_check = False

        cpa_index = self.params.index('cpa_actual')
        cpa_target_index = self.params.index('cpa_target')

        for li in data:

            #Check if Target CPA is one of the considered parameters
            if target_check:
                new_lineitem = np.zeros(nfeatures - 1)
            else:
                new_lineitem = np.zeros(nfeatures)

            for feature_index in range(nfeatures):
                feature_value = li[self.params[feature_index]]

                if feature_value is None or feature_value < 0:
                    if feature_index == cpa_target_index:
                        new_lineitem[feature_index] = False
                    else:
                        invalidata.append(li)
                    break
                else:

                    if self.params[feature_index] == 'cpa_target':
                        cpa_target = feature_value
                    else:
                        new_lineitem[feature_index] = feature_value
            else:
                if target_check:

                    if not new_lineitem[cpa_target_index]: # If the CPA target is missing
                        new_lineitem[cpa_index] = -1000 # Set missing CPA to -infinity to close the Deep Learner input gates
                    else:
                        new_lineitem[cpa_index] -= cpa_target
                validata.append(li)
                matrix_data.append(new_lineitem)

        return matrix_data, validata, invalidata

    def normalize_training_data(self, data, method='min-max', pacing_positions=False, CPA_target_pos=False):

        """
        :param data: Processed Data or Data class array
        :param method: Method to use in the normalization. (Default: 'min-max', alternative: 'z-norm')
        :return: normalize_training_data Data
        """
        norm_matrix = []
        ncases = len(data)

        for case_set_index in range(ncases):
            case_matrix = np.array(data[case_set_index]).astype(float)
            norm_case = self.normalize_data(case_matrix, method, pacing_positions, CPA_target_pos, self.normalizing_factor)
            norm_matrix.append(norm_case)
        return norm_matrix

    @staticmethod
    def whole_data_statistics_norm(data, method='min-max', pacing_positions=False, CPA_target_pos=False):

        """
        Normalization using the whole data statistics instead of intra case statistics
        :param data: Line item data: [case1, case2, ..., caseN]
        :param method: Method for normalization 'min-max' or 'z-norm'
        :param pacing_positions: If normalizing Pacing around 100 pass in the position of the pacing metrics
        :param CPA_target_pos: if normalizing CPA to CPAtarget ratio around 1, pass in the position of the metric
        :return: Normalized data
        """

        norm_data = np.array(data)
        flatten_data = np.vstack(norm_data)
        nfeatures = np.shape(flatten_data)[-1]
        if method == 'z-norm':
            means = np.zeros((nfeatures))
            for i in range(len(means)):
                if pacing_positions and i in pacing_positions:
                    means[i] = 100
                elif CPA_target_pos and i == CPA_target_pos:
                    means[i] = 1
                else:
                    means[i] = np.mean(flatten_data[:, i], axis=0)
            stds = np.std(flatten_data, axis=0)
        elif method =='min-max':
            min_values = np.min(flatten_data, axis=0)
            max_values = np.max(flatten_data, axis=0)
            max_min_diffs = max_values - min_values

        norm_data = np.array(data)
        ncases = len(data)
        for case_index in range(ncases):
            case = np.array(data[case_index])
            for i in range(nfeatures):
    
                if method == 'min-max':
                    if pacing_positions and i in pacing_positions:
    
                            case[:, i] = (case[:, i] - 100) / \
                                              (max_min_diffs[i])
    
                    elif CPA_target_pos and i == CPA_target_pos:
    
                        if method == 'min-max':
    
                            case[:, i] = (case[:, i] - 0) / \
                                              max_min_diffs[i]
                    else:
                        if max_min_diffs[i] == 0:
                            case[:, i] = np.ones(np.shape(case[:, i]))
                        else:
                            case[:, i] = (case[ :, i] - min_values[i]) / \
                                              (max_min_diffs[i])
    
                elif method == 'z-norm':
                        case[:, i] = (case[ :, i] - means[i]) / stds[i]
            norm_data[case_index] = case

        return norm_data

    @staticmethod
    def normalize_data(data, method='min-max',  pacing_positions=False, CPA_target_pos=False, normalizing_factor=0):
        """
        Normalization of data using statistics of each case individually
        :param data: Matrix of data
        :param method: method for normalization, min-max or z-norm
        :param pacing_positions: if normalizing pacing around 100, pass a list with the pacing positions
        :param CPA_target_pos: Same as pacing_positions but to set the normalization of CPA-CPA_target around 0
        :param normalizing_factor: If an offset is desired pass in the offset value
        :return: normalized_data
        """

        norm_data = np.array(data)
        nfeatures = np.shape(data)[-1]

        for i in range(nfeatures):

            if pacing_positions and i in pacing_positions:
                if method == 'min-max':

                    norm_data[:, i] = (norm_data[:, i] - 100) / \
                                      (np.max(norm_data[:, i]) - np.min(norm_data[:, i])) + normalizing_factor
                elif method == 'z-norm':
                    std_ = np.std(norm_data[:, i])
                    if std_ == 0:
                        std_ = np.mean(norm_data[:, i])
                    norm_data[:, i] = (norm_data[:, i] - 100) / \
                                      (std_)
            elif CPA_target_pos and i == CPA_target_pos:

                if method == 'min-max':

                    norm_data[:, i] = (norm_data[:, i] - 0) / \
                                      (np.max(norm_data[:, i]) - np.min(norm_data[:, i])) + normalizing_factor
                elif method == 'z-norm':
                    norm_data[:, i] = (norm_data[:, i] - 1) / \
                                      (np.std(norm_data[:, i]))

            else:

                if method == 'z-norm':
                    std = np.std(norm_data[:, i])
                    if std == 0:
                        std = 1
                    norm_data[:, i] = (norm_data[:, i] - np.mean(norm_data[:, i])) / std
                elif method == 'min-max':
                    if -np.inf in norm_data[:,i]:
                        valid_cpa_pos = norm_data[:,i] != -np.inf
                        invalid_cpa_pos = norm_data[:,i] == -np.inf
                        norm_data[valid_cpa_pos, i] = (norm_data[valid_cpa_pos, i] - np.min(norm_data[valid_cpa_pos, i])) / \
                                          (np.max(norm_data[valid_cpa_pos, i]) - np.min(norm_data[valid_cpa_pos, i])) + normalizing_factor
                        norm_data[invalid_cpa_pos,i] = -100

                    elif np.max(norm_data[:, i]) == np.min(norm_data[:, i]):
                        norm_data[:, i] = np.ones(np.shape(norm_data[:, i]))
                    else:
                        norm_data[:, i] = (norm_data[:, i] - np.min(norm_data[:, i])) / \
                                          (np.max(norm_data[:, i]) - np.min(norm_data[:, i])) + normalizing_factor
        return norm_data