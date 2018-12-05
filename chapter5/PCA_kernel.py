from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation

    Parameters
    ---------
    :param X:{Numpy ndarray}, shape = [n_samples, n_features]
    :param gamma: float
        Tuning parameter of the RBF kernel
    :param n_components: int
        Number of principal components to return
    :return X_pc:{Numpy ndarray}, shape = [n_samples, k_features]
         Projected dataset
    """
    #Calculate pairwise squared Euclidean distances
    #in the MxN dimensional dataset
    sq_disks = pdist(X, 'sqeuclidean')

    #Convert pairwise distances into a square matrix
    mat_sq_dists = squareform(sq_disks)

    #Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    #Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    #Obtaining eigenpairs from the centered kernel matrix
    #numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    #Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
    return X_pc

#分离半月形数据
def split_moons():
    X, y = make_moons(n_samples=100, random_state=123)
    # plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    # plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    # plt.show()
    #应用ＰＣＡ将数据映射到主成分上
    # scikit_pca = PCA(n_components=2)
    # X_spca = scikit_pca.fit_transform(X)

    #应用核PCA实现
    X_spca = rbf_kernel_pca(X, gamma=15, n_components=2)

    #PLT显示
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.show()

#分离同心圆
def split_crcly():
    X,y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
    # plt.scatter(X[y==0, 0], X[y==0, 1], color='red', marker='^', alpha=0.5)
    # plt.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='o', alpha=0.5)
    # plt.show()

    #使用标准PCA
    # scikit_pca = PCA(n_components=2)
    # X_spca = scikit_pca.fit_transform(X)

    #基于核PCA
    X_spca = rbf_kernel_pca(X, gamma=15, n_components=2)

    #PLT显示
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
    ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y==0, 0], np.zeros((500,1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y==1, 0], np.zeros((500,1))-0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.show()

if __name__ == '__main__':
    split_crcly()
