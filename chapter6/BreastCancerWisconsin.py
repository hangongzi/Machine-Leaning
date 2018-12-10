import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, validation_curve, GridSearchCV
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import roc_curve, auc
from scipy import interp

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

def pca_scl():
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1))])
    pipe_lr.fit(X_train, y_train)
    print('Test Accuracy: %.3f'%pipe_lr.score(X_test, y_test))

def k_zhe():
    kfold = StratifiedKFold(n_splits=10,random_state=1)
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1, solver='lbfgs'))])
    #Ascores = []
    for train_index, test_index in kfold.split(X, y):
        #print('TRAIN:', train_index, '\nTEST:', test_index)
        X_train_skf, X_test_skf = X[train_index], X[test_index]
        y_train_skf, y_test_skf = y[train_index], y[test_index]
        pipe_lr.fit(X_train_skf, y_train_skf)
        score = pipe_lr.score(X_test_skf, y_test_skf)
        #scores.append(score)
        #print('Fold: %s, class disk.: %s, ACC: %.3f' % (train_index, y_train_skf, score))
        print('ACC: %.3f' % score)
    #print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    #交叉评估
    scores_c = cross_val_score(estimator=pipe_lr, X=X_train,y=y_train,cv=10,n_jobs=1)
    print('CV accuracy scores:%s'% scores_c)
    print('CV accuracy:%.3f +/- %.3f'%(np.mean(scores_c),np.std(scores_c)))

#学习曲线
def learn_fun():
    pipe_lr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(random_state=0, solver='lbfgs'))])
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean + test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()

#验证曲线
def val_fun():
    pipe_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=0))])
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train, y=y_train, param_name='clf__C',
                                                 param_range=param_range, cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.mean(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.show()


def grid_search_and_cross_val():
    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC(random_state=1))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C':param_range,
                   'clf__kernel':['linear']},
                  {'clf__C': param_range,
                   'clf__gamma':param_range,
                   'clf__kernel':['rbf']}]
    # 网格搜索(通过超参数值的改动对机器学习模型进行调优)
    # gs = GridSearchCV(estimator=pipe_svc,
    #                   param_grid=param_grid,
    #                   scoring='accuracy',
    #                   cv=10,
    #                   n_jobs=-1)
    # gs =gs.fit(X_train,y_train)
    # # print(gs.best_score_)
    # # print(gs.best_params_)
    # clf = gs.best_estimator_
    # clf.fit(X_train, y_train)
    # print('Test accuracy: %.3f'%clf.score(X_test, y_test))

    #嵌套交叉验证（在不同机器学习算法中作出选择）
    # gs_cross = GridSearchCV(estimator=pipe_svc,
    #                         param_grid=param_grid,
    #                         scoring='accuracy',
    #                         cv=10,
    #                         n_jobs=-1)
    # scores = cross_val_score(gs_cross, X, y, scoring='accuracy', cv=5)
    # print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    #嵌套交叉验证（决策树）
    # gs_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
    #                        param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
    #                        scoring='accuracy',
    #                        cv=5)
    # scores = cross_val_score(gs_tree, X_train, y_train, scoring='accuracy',
    #                          cv=5)
    # print('CV accuracy: %.3f +/- %.3f'%(np.mean(scores), np.std(scores)))

    #自定义评分
    scorer = make_scorer(f1_score, pos_label=0)
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring=scorer,
                      cv=10)

def read_confuse_metix():
    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC(random_state=1))])
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    #print(confmat)
    # fig, ax = plt.subplots(figsize=(2.5, 2.5))
    # ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(confmat.shape[0]):
    #     for j in range(confmat.shape[1]):
    #         ax.text(x=j, y=i,
    #                 s=confmat[i, j],
    #                 va='center', ha='center')
    # plt.xlabel('predicted label')
    # plt.ylabel('true label')
    # plt.show()

    #其他性能指标
    # print('Recall: %.3f'% recall_score(y_true=y_test, y_pred=y_pred))
    # print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

#收式者特征曲线
def roc():
    X_train2 = X_train[:, [4, 14]]
    kfold = StratifiedKFold(n_splits=3, random_state=1)
    #cv = kfold.split(X_train, y_train)
    fig = plt.figure(figsize=(7,5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1, solver='lbfgs'))])
    count_cv = 0
    for train, test in kfold.split(X_train, y_train):
        probas = pipe_lr.fit(X_train2[train],
                             y_train[train]).predict_proba(X_train2[test])
        fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr,
                 tpr,
                 lw=1,
                 label='ROC fold %d (area = %0.2f)' % (count_cv+1, roc_auc))
        count_cv += 1

    plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
    mean_tpr /= count_cv
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 0, 1],
             [0, 1, 1],
             lw=2,
             linestyle=':',
             color='black',
             label='perfect performance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC(random_state=1))])
    pipe_svc = pipe_svc.fit(X_train2, y_train)
    y_pred2 = pipe_svc.predict(X_test[:, [4, 14]])
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import accuracy_score
    print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2))
    print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred2))

if __name__ == '__main__':
    roc()