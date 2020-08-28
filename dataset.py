# -*- coding：utf-8 -*-

import numpy as np
import pandas as pd
FILEPATH = './data/train.csv'
rng = np.random.RandomState(1)


def standard_age(data):
    mean = 29.70
    theta = 14.53
    data = (data - mean) / theta
    return data


def load_data(filepath=FILEPATH):
    ori_data = pd.read_csv(filepath)
    ori_data.replace(['female', 'male'], [0, 1], inplace=True)
    train_data = pd.concat([ori_data['Survived'],
                            ori_data['Pclass'],
                            ori_data['Sex'],
                            ori_data['Age'],
                            ori_data['SibSp'],
                            ori_data['Parch']],
                           axis=1)
    for index, row in train_data.iterrows():
        if np.isnan(row['Age']):
            if row['Pclass'] == 1.0 and row['Sex'] == 1.0:
                train_data.loc[index, 'Age'] = 40.9
            elif row['Pclass'] == 1.0 and row['Sex'] == 0.0:
                train_data.loc[index, 'Age'] = 33.3
            elif row['Pclass'] == 2.0 and row['Sex'] == 1.0:
                train_data.loc[index, 'Age'] = 31.0
            elif row['Pclass'] == 2.0 and row['Sex'] == 0.0:
                train_data.loc[index, 'Age'] = 29.2
            elif row['Pclass'] == 3.0 and row['Sex'] == 1.0:
                train_data.loc[index, 'Age'] = 26.8
            else:
                train_data.loc[index, 'Age'] = 22.7
        else:
            pass
    train_data['Age'] = standard_age(train_data['Age'])

    # train_p = np.asarray(train_data[train_data['Survived'] == 1])
    # train_n = np.asarray(train_data[train_data['Survived'] == 0])
    # sample_len = min(len(train_p), len(train_n))
    # if len(train_p) is sample_len:
    #     np.random.shuffle(train_n)
    # else:
    #     np.random.shuffle(train_p)
    # dataset = np.vstack((train_p[0:sample_len], train_n[0:sample_len]))

    dataset = np.asarray(train_data)
    np.random.shuffle(dataset)
    return dataset
