from sklearn.ensemble import RandomForestClassifier
from chapter3 import sigmoid
import numpy as np
import matplotlib.pyplot as plt

def train_forest():
    X, y, X_train, X_test, y_train, y_test, X_train_std, X_test_std = sigmoid.data_set()
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=0, n_jobs=8)
    forest.fit(X_train,y_train)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    sigmoid.plot_decision_regions(X_combined_std,y_combined,classifier=forest,test_idx=range(105,150))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    train_forest()