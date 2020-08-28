# -*- coding:utf-8 -*-

import joblib
import numpy as np
import pandas as pd
FILEPATH = './data/test.csv'


def standard_age(data):
    mean = 29.70
    theta = 14.53
    data = (data - mean) / theta
    return data


data = pd.read_csv(FILEPATH)
data.replace(['female', 'male'], [0, 1], inplace=True)
test_data = pd.concat([data['PassengerId'],
                       data['Pclass'],
                       data['Sex'],
                       data['Age'],
                       data['SibSp'],
                       data['Parch']],
                      axis=1)

for index, row in test_data.iterrows():
    if np.isnan(row['Age']):
        if row['Pclass'] == 1.0 and row['Sex'] == 1.0:
            test_data.loc[index, 'Age'] = 40.9
        elif row['Pclass'] == 1.0 and row['Sex'] == 0.0:
            test_data.loc[index, 'Age'] = 33.3
        elif row['Pclass'] == 2.0 and row['Sex'] == 1.0:
            test_data.loc[index, 'Age'] = 31.0
        elif row['Pclass'] == 2.0 and row['Sex'] == 0.0:
            test_data.loc[index, 'Age'] = 29.2
        elif row['Pclass'] == 3.0 and row['Sex'] == 1.0:
            test_data.loc[index, 'Age'] = 26.8
        else:
            test_data.loc[index, 'Age'] = 22.7
    else:
        pass
test_data['Age'] = standard_age(test_data['Age'])
test_data = np.asarray(test_data)

test_data = np.asarray(test_data)
passenger_id = test_data[:, 0]
test_x = test_data[:, 1:]
model = joblib.load("./model/2020-08-28_10-05-11_titanic_pred.pkl")
test_y = model.predict(test_x)
result = pd.DataFrame({'PassengerId': [int(
    x) for x in passenger_id], 'Survived': [int(x) for x in test_y]})
result.to_csv("./data/submission.csv", index=False)
print("完成.")
