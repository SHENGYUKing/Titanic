# -*- coding:utf-8 -*-

import joblib
import numpy as np
import pandas as pd
FILEPATH = './data/test.csv'

data_csv = pd.read_csv(FILEPATH)
data = pd.DataFrame(data_csv)
data.replace(['female', 'male'], [0, 1], inplace=True)
test_data = pd.concat([data['PassengerId'], data['Pclass'], data['Sex'], data['Age']], axis=1)
test_data = np.asarray(test_data)
for line in test_data:
    if np.isnan(line[-1]):
        if line[2] is 'male':
            line[2] = 1
        else:
            line[2] = 0

        if line[1] is 1 and line[2] is 1:
            line[-1] = 40.9
        elif line[1] is 1 and line[2] is 0:
            line[-1] = 33.3
        elif line[1] is 2 and line[2] is 1:
            line[-1] = 31
        elif line[1] is 2 and line[2] is 0:
            line[-1] = 29.2
        elif line[1] is 3 and line[2] is 1:
            line[-1] = 26.8
        else:
            line[-1] = 22.7
passenger_id = test_data[:, 0]
test_x = test_data[:, 1:]
model = joblib.load("./model/titanic_pred(rng1,acc821839,LR,nopre).pkl")
test_y = model.predict(test_x)
result = pd.DataFrame({'PassengerId': [int(x) for x in passenger_id], 'Survived': [int(x) for x in test_y]})
result.to_csv("./data/submission.csv", index=False)
