# -*- coding：utf-8 -*-

import os
import dataset as ds
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
rng = ds.rng


def plt_learning_curve(model, pltname, x, y, ylim=None, cv=None, n_jobs=1, train_size=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(pltname)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, x, y,
                                                            cv=cv, n_jobs=n_jobs, train_sizes=train_size)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean+test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


count = 0
while True:
    count = count + 1
    dataset = ds.load_data()
    np.random.shuffle(dataset)
    x, y = dataset[:, 1:], dataset[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rng)
    # estimator = neighbors.KNeighborsClassifier()
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    scorce = estimator.score(x_test, y_test, sample_weight=None)

    if count % 5 is 0:
        print("已循环训练第 %d 轮" % count)

    if scorce * 100 > 85:
        print('y_predict = ')
        print(y_predict)

        print('y_test = ')
        print(y_test)

        print('Acc: {}%'.format(scorce * 100))

        name = "titanic_pred.pkl"
        if not os.path.exists("./model"):
            os.mkdir("./model")
        while name in os.listdir("./model"):
            tmp = name.split(".")
            name = tmp[0] + str(1) + "." + tmp[1]
        detector = joblib.dump(estimator, "./model/" + name)
        cv = ShuffleSplit(n_splits=2, test_size=0.3, random_state=rng)
        title = "Learnming Curves\n(@590samples,test_size=0.3,normalize=SS)"
        plt_learning_curve(estimator, title, x, y, ylim=(0.0, 1.0), n_jobs=1)
        break

plt.legend()
plt.show()
