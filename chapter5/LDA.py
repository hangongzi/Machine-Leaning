import numpy as np
from chapter5 import PCA
import matplotlib.pyplot as plt
from lda import LDA
from sklearn.linear_model import LogisticRegression

def LDS_define():
    np.set_printoptions(precision=4)
    mean_vecs = []
    X, y ,X_train_std, X_test_std, y_train, y_test = PCA.split_dataset()
    for label in range(1,4):
        mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    #    print('MV %s: %s\n' % (label, mean_vecs[label-1]))

    d = 13
    S_W = np.zeros((d,d))
    # for label, mv in zip(range(1,4), mean_vecs):
    #     class_scatter = np.zeros((d,d))
    #     for row in X[y == label]:
    #         row, mv = row.reshape(d,1), mv.reshape(d, 1)
    #         class_scatter += (row-mv).dot((row-mv).T)
    #     S_W += class_scatter

    #对散布矩阵做缩放处理：
    for label, mv in zip(range(1,4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter

    #print('Within-class scatterr matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
    #print('Class label distrubution: %s' % np.bincount(y_train)[1:])

    mean_overall = np.mean(X_train_std, axis=0)
    d = 13
    S_B = np.zeros((d, d))
    for i, mean_vec in enumerate(mean_vecs):
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(d, 1)
        mean_overall = mean_overall.reshape(d, 1)
    S_B += n*(mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
    #print('Between-class scatter matrix: %sx%s' % (S_B.shape[0],S_B.shape[1]))
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    #对特征值进行排序
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda  k:k[0], reverse=True)
    #print('Eigenvalues in decreasing order: \n')
    #for eigen_val in eigen_pairs:
    #    print(eigen_val[0])

    tot = sum(eigen_vals.real)
    discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)

    # plt.bar(range(1,14), discr, alpha=0.5, align='center', label='individual "discriminability"')
    # plt.step(range(1,14), cum_discr,where='mid', label='cumulative "discriminability"')
    # plt.ylabel('"discriminability" ratio')
    # plt.xlabel('Linear Discriminants')
    # plt.legend(loc='best')
    # plt.show()

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                   eigen_pairs[1][1][:, np.newaxis].real))
    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train==1, 0],
                    X_train_lda[y_train==1, 1],
                    c=c, label=l, marker=m)
    plt.xlabel('LD 1')
    plt.ylabel('LD 1')
    plt.legend(loc='upper right')
    plt.show()

def LDS_test():
    X, y, X_train_std, X_test_std, y_train, y_test = PCA.split_dataset()
    lda = LDA(n_topics=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)
    lr = LogisticRegression()
    lr = lr.fit(X_train_lda, y_train)
    PCA.plot_decision_regions(X_train_lda, y_train, classifier=lr)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='lower left')
    plt.show()

if __name__ == '__main__':
    LDS_test()