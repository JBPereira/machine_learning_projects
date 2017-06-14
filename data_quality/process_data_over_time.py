import os
import re
from datetime import datetime
import numpy as np


def get_size(start_path, import_account):
    """
    :param start_path: Path for the imported data folder
    :return: date_array, an array wityh the dates for each imported data set
             size_array, an array with the sizes for each imported data set
    """
    size_array = []
    date_array = []
    for dirpath, dirnames, filenames in os.walk(start_path):
        total_size = 0
        for f in filenames:
            if import_account in f:
                fp = os.path.join(dirpath, f)
                match = re.search(r'\d{4}/\d{1}/\d{2}', fp)
                if match is None:
                    match = re.search(r'\d{4}/\d{2}/\d{2}', fp)
                if match is None:
                    match = re.search(r'\d{4}/\d{1}/\d{1}', fp)
                if match is None:
                    match = re.search(r'\d{4}/\d{2}/\d{1}', fp)
                date = datetime.strptime(match.group(), '%Y/%m/%d').date()
                # str_date = date.strftime('%m/%d/%Y')
                date_array.append(date)
                with open(fp) as file:
                    for row, l in enumerate(file):
                        pass
                size = row + 1
                size_array.append(size)

    return date_array, size_array


def sort_by_time(date_array, size_array):
    """
    :param date_array: Array with the dates of the files
    :param size_array: Array with the sizes of the files per day
    :return: ordered date array and size array
    """

    date_ = []
    for date_index in range(len(date_array)):
        date_.append(date_array[date_index].timetuple().tm_yday)
    date_ = np.array(date_)
    date_array = np.array(date_array)
    size_array = np.array(size_array)

    arr_inds = date_.argsort()
    date_array = date_array[arr_inds]
    for date_index in range(len(date_array)):
        date_array[date_index] = date_array[date_index].strftime('%m/%d/%Y')
    sorted_size = size_array[arr_inds]
    sorted_size = np.array(sorted_size)
    date_ = date_[arr_inds]

    return date_, sorted_size