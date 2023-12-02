import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class svm_rbf():
    def __init__(self, data, label):
        self.name = 'SVM_RBF'
        self.trainx = data
        self.trainy = label

    def train(self):
        cost = []
        gamma = []
        for i in range(-3, 10, 2):
            cost.append(np.power(2.0, i))
        for i in range(-5, 4, 2):
            gamma.append(np.power(2.0, i))

        parameters = {'C': cost, 'gamma': gamma}
        svm = SVC(verbose=0, kernel='rbf')
        clf = GridSearchCV(svm, parameters, cv=3)
        clf.fit(self.trainx, self.trainy)

        # print(clf.best_params_)
        bestc = clf.best_params_['C']
        bestg = clf.best_params_['gamma']
        tmpc = [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0,
                0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cost = []
        gamma = []
        for i in tmpc:
            cost.append(bestc * np.power(2.0, i))
            gamma.append(bestg * np.power(2.0, i))
        parameters = {'C': cost, 'gamma': gamma}
        svm = SVC(verbose=0, kernel='rbf')
        clf = GridSearchCV(svm, parameters, cv=3)
        clf.fit(self.trainx, self.trainy)
        # print(clf.best_params_)
        p = clf.best_estimator_
        return p
