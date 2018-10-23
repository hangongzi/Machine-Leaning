from sklearn.neighbors import KNeighborsClassifier
from chapter3 import sigmoid
import numpy as np
import matplotlib.pyplot as plt

def knn_train():
    X, y, X_train, X_test, y_train, y_test, X_train_std, X_test_std = sigmoid.data_set()
    knn = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')
    knn.fit(X_train, y_train)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    sigmoid.plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.show()

if __name__ == '__main__':
    knn_train()
