import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def drop_nan():
    csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.0,11.0,12.0,'''
    #csv_data = unicode(csv_data)
    df = pd.read_csv(StringIO(csv_data))
    return df
    # #通过values方法访问df的Numpy数组
    # print(df.values)
    # #删除带有NaN值的行
    # print(df.dropna())
    # #将axis=1，删除数据集中至少包含一个NaN值的列
    # print(df.dropna(axis=1))
    # #只删除所有值是NaN的列的行
    # print(df.dropna(how='all'))
    # #删除有不是至少4个非NaN值的行
    # print(df.dropna(thresh=4))
    # #删除C列中含有NaN值的行
    # print(df.dropna(subset=['C']))

#通过插值删除NaN
def insert_nan():
    df = drop_nan()
    imr = SimpleImputer(strategy='median')
    imr =imr.fit(df)                #构建数据补齐模型
    imputed_data = imr.transform(df.values)#对数据集中相应的数据补齐
    print(imputed_data)

#处理数据的类别
def classfiy_data():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']])
    df.columns = ['color', 'size', 'price', 'classlabel']
    size_mapping = {
        'XL':3,
        'L':2,
        'M':1
    }
    df['size'] = df['size'].map(size_mapping)
    #逆映射字典
    inv_size_mapping = {v:k for k,v in size_mapping.items()}
    class_mapping = {
        label: idx for idx, label in enumerate(np.unique(df['classlabel']))
    }
    #列表转换
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    #df['classlabel'] = df['classlabel'].map(class_mapping)
    #df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    #y = class_le.inverse_transform(y)
    X = df[['color','size','price']].values
    X[:,0] = class_le.fit_transform(X[:,0])
    #使用独热编码技术
    ohe = OneHotEncoder(categorical_features=[0],sparse=False)
    #获取由独热编码技术生成的虚拟特征
    print(pd.get_dummies(df[['price','color','size']]))

#将数据集分为训练集与测试集
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
    return X_train, X_test, y_train, y_test ,df_wine

def peature_scal():
    X_train, X_test, y_train, y_test, df_wine = split_dataset()
    #归一化缩放（最大最小缩放）
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.fit_transform(X_test)

    #标准化缩放
    stdsc =StandardScaler()
    X_train_std  = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)

    #使用L1正则化实现数据稀疏
    # lr = LogisticRegression(penalty='l1', C=0.1)
    # lr.fit(X_train_std, y_train)
    # print('Training accuracy:', lr.score(X_train_std, y_train))
    # print('Test accurcacy:', lr.score(X_test_std, y_test))
    # print(lr.intercept_)
    # print(lr.coef_)
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    colors = ['blue', 'green', 'red', 'cyan',
              'magenta', 'yellow', 'black',
              'pink', 'lightgreen', 'lightblue',
              'gray', 'indigo', 'orange']
    weights, params = [],[]
    for c in np.arange(-4, 6):
        lr = LogisticRegression(penalty='l1', C=10.0**c, random_state=0)
        lr.fit(X_train_std, y_train)
        weights.append(lr.coef_[1])
        params.append(10.0**c)

    weights = np.array(weights)
    for column, color in zip(range(weights.shape[1]),colors):
        plt.plot(params, weights[:, column], label = df_wine.columns[column+1], color=color)
    plt.axhline(0, color='black', linestyle='--', linewidth=3)
    plt.xlim([10**(-5), 10**5])
    plt.ylabel('weight coefficiente')
    plt.xlabel('C')
    plt.xscale('log')
    plt.legend(loc='upper left')

    #ax.legent(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
    plt.show()

if __name__ == '__main__':
    peature_scal()

