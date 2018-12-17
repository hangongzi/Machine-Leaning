import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from chapter10.LRGD import LinearRegressionGD
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

def dataset():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                     header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    # #print(df.head())
    # sns 显示
    # sns.set(style='whitegrid', context='notebook')
    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    # sns.pairplot(df[cols], height=2.5)
    # sns.reset_orig()
    # plt.show()
    # cm = np.corrcoef(df[cols].values.T)
    # sns.set(font_scale=1.5)
    # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
    # plt.show()

    X = df[['RM']].values
    y = df['MEDV'].values
    # sc_x = StandardScaler()
    # sc_y = StandardScaler()
    # X_std = sc_x.fit_transform(X)
    # y_std = sc_y.fit_transform(np.array(y).reshape(-1, 1))
    # lr = LinearRegressionGD()
    # lr.fit(X_std, y)
    # plt.plot(range(1, lr.n_iter+1), lr.cost_)
    # plt.ylabel('SSE')
    # plt.xlabel('Epoch')
    # plt.show()
    # lin_regplot(X_std, y, lr)
    # plt.xlabel('Average number of roons [RM] (standardized)')
    # plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    # plt.show()
    # x_arr = np.array([0.5]).reshape(-1,1)
    # num_rooms_std = sc_x.transform(np.array([5.0]).reshape(1, -1))
    # price_std = lr.predict(num_rooms_std)
    # print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))
    # print("Slope: %.3f" % lr.w_[1])
    # print("Intercept: %.3f" % lr.w_[0])

    # #python 库实现高效线性回归
    # slr = LinearRegression()
    # slr.fit(X, y)
    # # print('Slope: %.3f' % slr.coef_[0])
    # # print('Intercept: %.3f' % slr.intercept_)
    # lin_regplot(X, y, slr)
    # plt.xlabel('Average number of rooms [RM] (standardized)')
    # plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    # plt.show()
    ransac_fun(X, y)


def ransac_fun(X, y):
    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=50,
                             residual_threshold=5.0,
                             random_state=0)
    ransac.fit(X, y=y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(np.array(line_X[: np.newaxis]).reshape(-1,1))
    # plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
    # plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
    # plt.plot(line_X, line_y_ransac, color='red')
    # plt.xlabel('Average number of rooms [RM]')
    # plt.ylabel('Price in $1000\'s [MEDV]')
    # plt.legend(loc='upper left')
    # plt.show()
    print('Slope: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

def val_model():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                     header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)

    #绘制残差图（将预测结果减去对应目标变量真实值）
    # plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
    # plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    # plt.xlabel('Predicted values')
    # plt.ylabel('Residuals')
    # plt.legend(loc='upper left')
    # plt.hlines(y=0, xmin=-10, xmax=50, lw=2, colors='red')
    # plt.xlim([-10, 50])
    # plt.show()

    #均方误差（MSE）
    # print('MSE train: %.3f, test: %.3f'%
    #      (mean_squared_error(y_train, y_train_pred),
     #      mean_squared_error(y_test, y_test_pred)))

    # r2
    # print('R^2 train: %.3f, test: %.3f' %
    #       (r2_score(y_train, y_train_pred),
    #        r2_score(y_test, y_test_pred)))

    # # 线性回归的正则化
    # ridge = Ridge(alpha=1.0) # 岭回归
    # lasso = Lasso(alpha=1.0)
    # lasso = ElasticNet(alpha=1.0, l1_ratio=0.5)

def duoxiangshi():
    X = np.array([258.0, 270.0, 294.0,
                  320.0, 342.0, 368.0,
                  396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
    y = np.array([236.4, 234.4, 252.8,
                  298.6, 314.2, 342.2,
                  360.8, 368.0, 391.2, 390.8])
    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)
    lr.fit(X, y)
    X_fit = np.arange(250, 600, 10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)
    pr.fit(X=X_quad, y=y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
    # plt.scatter(X, y, label='training points')
    # plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
    # plt.plot(X_fit, y_quad_fit, label='quadratic fit')
    # plt.legend(loc='upper left')
    # plt.show()
    y_lin_pred = lr.predict(X)
    y_quad_pred = pr.predict(X_quad)
    print('Training MSE linear: %.3f, quadratic: %.3f'%(
        mean_squared_error(y, y_lin_pred),
        mean_squared_error(y, y_quad_pred)
    ))
    print('Training R^2 linear: %.3f, quadratic: %.3f' % (
        r2_score(y, y_lin_pred),
        r2_score(y, y_quad_pred)
    ))

def duoxiangshi_houses():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                     header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    X = df[['LSTAT']].values
    y = df['MEDV'].values

    regr = LinearRegression()
    #
    # #crate polynominal features
    # quadratic = PolynomialFeatures(degree=2)
    # cubic = PolynomialFeatures(degree=3)
    # X_quad = quadratic.fit_transform(X)
    # X_cubic = cubic.fit_transform(X)
    #
    # #linear fit
    # X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    # regr = regr.fit(X, y)
    # y_lin_fit = regr.predict(X_fit)
    # linear_r2 = r2_score(y, regr.predict(X))
    #
    # #quadratic fit
    # regr = regr.fit(X_quad, y)
    # y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    # quadratic_r2 = r2_score(y, regr.predict(X_quad))
    #
    # # cubic fit
    # regr = regr.fit(X_cubic, y)
    # y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    # cubic_r2 = r2_score(y, regr.predict(X_cubic))
    #
    # #plot results
    # plt.scatter(X, y, label='training points', color='lightgray')
    # plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$'%linear_r2, color='blue', lw=2, linestyle=':')
    # plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$'%quadratic_r2, color='red', lw=2, linestyle='-')
    # plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$'%cubic_r2, color='green', lw=2, linestyle='--')
    # plt.xlabel('% lower status of the population [LSTAT]')
    # plt.ylabel('Price in $1000\'s [MEDV]')
    # plt.legend(loc='upper right')
    # plt.show()

    # 将LSTAT的对数值及MEDV的平方根映射到一个线性特征空间
    # transform features
    X_log = np.log(X)
    y_sqrt = np.sqrt(y)
    # fit features
    X_fit = np.arange(X_log.min() - 1,
                      X_log.max() + 1, 1)[:, np.newaxis]
    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

    plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
    plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2)
    plt.xlabel('log(% lower status of the population [LSTAT])')
    plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == '__main__':
    duoxiangshi_houses()