# -*- coding：utf-8 -*-

import numpy as np
import pandas as pd
FILEPATH = './data/train_prepro.csv'
rng = np.random.RandomState(1)


def load_data(filepath=FILEPATH):
    data_csv = pd.read_csv(filepath)
    data = pd.DataFrame(data_csv)
    # data['Pclass'] = (data['Pclass'] - 1)/2
    # data['Age'] = data['Age']/29.3
    data_positive = data[data['Survived'] == 1]
    data_negative = data[data['Survived'] == 0].sample(n=len(data_positive), random_state=rng)
    dataset = pd.concat([data_positive, data_negative]).values

    return dataset
