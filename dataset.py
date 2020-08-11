# -*- coding：utf-8 -*-

import numpy as np
import pandas as pd
import random
FILEPATH = './data/train.csv'
rng = np.random.RandomState(1)


def standard_age(data):
    mean = 29.70
    theta = 14.53
    data = (data - mean) / theta
    return data


def load_data(filepath=FILEPATH):
    data_csv = pd.read_csv(filepath)
    data = pd.DataFrame(data_csv)
    data.replace(['female', 'male'], [0, 1], inplace=True)
    train_data = pd.concat([data['Survived'], data['Pclass'], data['Sex'], data['Age'], data['SibSp'], data['Parch']], axis=1).dropna()
    train_data = np.asarray(train_data)
    for line in train_data:
        line[0] = int(line[0])
        line[1] = line[1]
        line[2] = int(line[2])
        line[3] = standard_age(line[3])
        line[4] = int(line[4])
        line[5] = int(line[5])

    data_positive = np.array([line for line in train_data if line[0] == 1])
    data_negative = np.array([line for line in train_data if line[0] == 0])
    np.random.shuffle(data_positive)
    np.random.shuffle(data_negative)
    sample_len = max(len(data_positive), len(data_negative))
    dataset = np.vstack((data_positive[0:sample_len], data_negative[0:sample_len]))

    return dataset
