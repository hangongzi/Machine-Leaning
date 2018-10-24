from sklearn.ensemble import RandomForestClassifier
from chapter4 import Value_NaN
import numpy as np
import matplotlib.pyplot as plt

def randomforest_importances():
    X_train, X_test, y_train, y_test, df_wine = Value_NaN.split_dataset()
    feat_labels = df_wine.columns[1:]
    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    #for f in range(X_train.shape[1]):
     #   str = "%2d %-*s %f" % (f+1, 30, feat_labels[f], importances[indices[f]])

    # plt.title('Feature Importances')
    # plt.bar(range(X_train.shape[1]),
    #         importances[indices],
    #         color='lightblue',
    #         align='center')
    # plt.xticks(range(X_train.shape[1]),
    #            feat_labels, rotation=90)
    # plt.xlim([-1, X_train.shape[1]])
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    randomforest_importances()