#主成分分析前四个步骤：数据标准化、构造协方差矩阵、获得协方差矩阵的特征值和特征向量
#以及按降序排列特征值序列对应的特征向量
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def split_dataset():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
    df_wine.columns = ['Class label', 'Alcohol',
                       'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium',
                       'Total phenols', 'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity', 'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
    #print('Class labels', np.unique(df_wine['Class label']))
    #print(df_wine.head())
    X,y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
    X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)
    return X,y, X_train_std, X_test_std, y_train, y_test

def feature_select():
    X_train_std, X_test_std, y_train = split_dataset()
    cov_mat = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    # plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
    # plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
    # plt.xlabel('Principal components')
    # plt.ylabel('Explained variance ratio')
    # plt.legend(loc='best')
    # plt.show()
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))#选取主成分
    X_train_pca = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train),colors, markers):
        plt.scatter(X_train_pca[y_train==1, 0],
                    X_train_pca[y_train==1, 1],
                    c=c, label=1, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.xlim(-5,10)
    plt.ylim(-3,6)
    plt.legend(loc='lower left')
    plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() -1, X[:, 0].max() +1
    x2_min, x2_max = X[:,1].min() -1, X[:, 1].max() +1
    #x3_min, x3_max = X[:,2].min() -1, X[:, 2].max() +1
    xx1, xx2, xx3 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    #Z = classifier.predict(X)
    #Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

def LogsticregressionPredict():
    X_train_std, X_test_std, y_train, y_test = split_dataset()
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    #print(pca.explained_variance_ratio_)
    X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
    LogsticregressionPredict()
