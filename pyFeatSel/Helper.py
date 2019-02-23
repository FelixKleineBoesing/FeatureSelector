import pandas as pd
import numpy as np
from itertools import chain, combinations


def create_k_fold_indices(len_data: int, k: int):
    assert type(len_data) == int
    assert type(k) == int
    assert k > 0
    assert len_data > 0
    assert len_data > k
    '''
    custom function to create k fold subsets
    :param len_data: number of observations in data
    :param k: number of folds
    :return: return list of dictionaries which contain train, test and validation loc indices
    '''

    indices = np.arange(len_data)

    indices_test = np.arange(len_data)
    np.random.shuffle(indices_test)

    indices_val = np.arange(len_data)
    np.random.shuffle(indices_val)

    test_groups = [indices_test[int(i*min((len_data/k), 0.2*len_data)):int((i+1)*min(len_data/k, 0.2*len_data))]
                   for i in range(k)]

    val_groups = []
    # shift test indices
    for j in range(len(test_groups)):
        if j == 0:
            val_groups += [test_groups[len(test_groups)-1]]
        else:
            val_groups  += [test_groups[j-1]]

    result = []
    for i in range(k):
        test = test_groups[i].tolist()
        val = val_groups[i].tolist()
        train = [int(i) for i in indices if i not in test + val]
        result += [{"test": test, "train": train, "val": val}]

    return result


def create_all_combinations(val: list):
    assert type(val) == list
    assert len(val) > 0
    '''
    createa all possible combinations from a list of values
    :param ss:
    :return:
    '''
    return chain(*map(lambda x: combinations(val, x), range(0, len(val) + 1)))

if __name__=="__main__":
    indices = create_k_fold_indices(10, 2)
    print(indices)