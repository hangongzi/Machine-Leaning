from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from sklearn.decomposition import KernelPCA

def rbf_kernel_pca(X, gamma, n_commpoents):
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
         lambdas: list
         Eigenvalues
    """
    #Calculate pairwise squared Euclidean distances
    #in the MxN dimensional dataset
    sq_dists = pdist(X, 'sqeuclidean')

    #Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    #Compute the symmetric kernel matrix
    K = exp(-gamma * mat_sq_dists)

    #Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    #obtaining eigenpairs from the centered kernel matrix
    #numpy.eigh returns them in sorted order
    eigvals, eigvecs = eigh(K)

    #Collect the top k eigenvectors (projected samples)
    alphas = np.column_stack((eigvecs[:, -i] for i in range(1, n_commpoents+1)))

    #Collect the corresponding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_commpoents+1)]
    return alphas, lambdas

def split_moons():
    X, y = make_moons(n_samples=100, random_state=123)
    alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_commpoents=1)
    x_new = X[25]
    X_proj = alphas[25]
    def project_x(x_new, X, gamma, alphas, lambdas):
        pair_disk = np.array([np.sum((x_new-row)**2) for row in X])
        k = np.exp(-gamma*pair_disk)
        return k.dot(alphas/lambdas)
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    plt.scatter(alphas[y==0, 0], np.zeros((50)), color='red', marker='^', alpha=0.5)
    plt.scatter(alphas[y==1, 0], np.zeros((50)), color='blue', marker='o', alpha=0.5)
    plt.scatter(X_proj, 0, color='black', label='original projection of point X[25]', marker='^', s=100)
    plt.scatter(x_reproj, 0, color='green', label='remapped point X[25]', marker='x', s=500)
    plt.legend(scatterpoints=1)
    plt.show()

def split_moons_kerpca():
    X, y = make_moons(n_samples=100, random_state=123)
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)
    plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

if __name__ == '__main__':
    split_moons_kerpca()