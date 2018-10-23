from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from chapter4 import SBS,Value_NaN

def train_sbs():
    X_train, X_test, y_train, y_test, df_wine = Value_NaN.split_dataset()
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS.SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    k_feat = [len(k) for k in sbs.subsets_]
    k5 = list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])
    knn.fit(X_train_std[:,k5],y_train)
    print('Training accuracy:', knn.score(X_train_std[:,k5], y_train))
    print('Test accuracy:', knn.score(X_test_std[:,k5], y_test))
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    train_sbs()
