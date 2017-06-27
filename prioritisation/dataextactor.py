import json
import os
import numpy as np
from data_sampling import stratified_sampling
from scipy.spatial import distance

"""
Package to save scrapped line item data (json format) into a csv file. Purpose: Get data for training or testing
prioritisation algorithm
"""


def save_li_data(line_item_data, filename_cpa, filename_wout_cpa):

    network_budget = line_item_data.get('config').get('budget').get('network')
    days_left = line_item_data.get('config').get('date_range').get('days_remaining')
    pacing_yesterday = line_item_data.get('data').get('yesterday').get('pacing').get('actual')
    pacing_todate = line_item_data.get('data').get('to_date').get('pacing').get('actual')
    primary_goal = str(line_item_data.get('config').get('primary_goal').get('id'))

    network_cpa = None
    cpa_target = None

    if primary_goal:
        goal = line_item_data.get('data').get('to_date').get('goals').get(primary_goal)
        if goal:
            network_cpa = line_item_data.get('data').get('to_date').get('goals').get(primary_goal).get('data').get('network_cpa').get('actual')
            cpa_target = line_item_data.get('data').get('to_date').get('goals').get(primary_goal).get('data').get('network_cpa').get('target')
    if None in (network_budget, days_left, pacing_yesterday, pacing_todate):
        return
    elif 0 in (network_budget, days_left, pacing_yesterday, pacing_todate, primary_goal):
        return
    elif pacing_yesterday < 0:
        return
    elif None in (network_cpa, cpa_target):
        data_file = open(filename_wout_cpa, "a")
        str_to_print = '{network_budget}, {days_left}, {pacing_yesterday}, {pacing_to_date}\n'
        data_file.write(str_to_print.format(network_budget=network_budget,
                                            days_left=days_left, pacing_yesterday=pacing_yesterday,
                                            pacing_to_date=pacing_todate))
        data_file.close()
    else:
        str_to_print = '{network_budget}, {days_left}, {pacing_yesterday}, {pacing_to_date}, ' \
                       '{network_cpa}, {cpa_target}\n'
        print str_to_print.format(network_budget=network_budget, days_left=days_left, pacing_yesterday=pacing_yesterday,
                                  pacing_to_date=pacing_todate, cpa_target=cpa_target,network_cpa=network_cpa)

        data_file = open(filename_cpa, "a")
        data_file.write(str_to_print.format(network_budget=network_budget,
                                            days_left=days_left, pacing_yesterday=pacing_yesterday,
                                            pacing_to_date=pacing_todate, cpa_target=cpa_target,
                                            network_cpa=network_cpa))
        data_file.close()


def randomize_days_left(data, dl_position = 1, budget_position=0):

    """
    Function to randomize the Days Left from one day data scrapping to get more variety out of this parameter.
    Non-uniform randomization of bigger campaign, i.e. to campaigns with larger budget it is assigned a random
    number of days left with higher probability for the range 1-30 and less for the range 30-365

    :param data: Data with line items
    :param dl_position: position of the Days Left
    :param budget_position:
    :return:
    """

    monthly_campaign_dl = np.arange(1, 30)
    ongoing_dl = np.arange(30, 365)

    non_uniform_dl_distribution = np.concatenate((np.repeat(monthly_campaign_dl, 15), ongoing_dl))
    mean_budget = np.mean(data[:, budget_position])
    for i in range(len(data)):
        if data[i, budget_position] > mean_budget:
            data[i, dl_position] = np.random.choice(non_uniform_dl_distribution, 1)
        else:
            data[i, dl_position] = np.random.choice(monthly_campaign_dl, 1)

    return data


def save_baas_json_to_csv(data_name, filename, filename_wout_cpa):

    with open(data_name) as json_file:
        data = json.load(json_file)

    try:
        os.remove(filename)
        os.remove(filename_wout_cpa)
    except OSError:
        pass

    save_li_data(data, filename_cpa=filename, filename_wout_cpa=filename_wout_cpa)




##select_case('BM - Search')
w_cpa = np.genfromtxt('BM-search2_data', delimiter=',')
w_cpa = np.vstack((w_cpa, np.genfromtxt('BM-search4_data', delimiter=',')))
w_cpa = np.vstack((w_cpa, np.genfromtxt('BM-search3_data', delimiter=',')))
w_cpa = np.vstack((w_cpa, np.genfromtxt('BM - Search', delimiter=',')))
wout_cpa = np.genfromtxt('BM-search2_wout_cpa', delimiter=',')
wout_cpa = np.vstack((wout_cpa, np.genfromtxt('BM-search4_wout_cpa', delimiter=',')))
wout_cpa = np.vstack((wout_cpa, np.genfromtxt('BM-search3_wout_cpa', delimiter=',')))
#
# w_cpa = randomize_days_left(w_cpa)
# wout_cpa = randomize_days_left(wout_cpa)
#
#
# w_cpa, wout_cpa = separate_data_type(li_data)


##################### Generate questionnaire data examples #########################

# examples_w_cpa = stratified_sampling(w_cpa, 10)
# examples_wout_cpa = stratified_sampling(wout_cpa[:, :4], 5)
#
# nan_array = np.zeros(len(examples_wout_cpa)) + np.nan
# nan_array = np.matrix(nan_array)
# examples_wout_cpa = np.append(examples_wout_cpa, nan_array.T, axis=1)
# examples_wout_cpa = np.append(examples_wout_cpa, nan_array.T, axis=1)
#
# examples = np.append(examples_w_cpa, examples_wout_cpa, axis=0)
# np.random.shuffle(examples)
#
# print 'done'
#
#
# print "Case 1"
# for line in examples:
#     case = []
#     for value_index in range(np.shape(line[0])[1]):
#         if value_index == 2 or value_index == 3:
#             case.append(str(float(line[0, value_index])/100).replace('.', ','))
#         else:
#             case.append(str(float(np.array(line[0, value_index]))).replace('.', ','))
#     print "{} \t {}\t {}\t {}\t {}\t {}".format(*case)
# print "\n"



#
# print "Case 2"
# for line in examples_two:
#     case = []
#     for value_index in range(len(line)):
#         if value_index == 2 or value_index == 3:
#             case.append(str(float(line[value_index])/100).replace('.', ','))
#         else:
#             case.append(str(float(line[value_index])).replace('.', ','))
#     print "{} \t {}\t {}\t {}\t {}\t {}".format(*case)
# print "\n"





