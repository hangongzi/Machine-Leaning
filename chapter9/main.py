from chapter8.class_exetor_mem import class_fun, stop
import pickle
import os

if __name__ == '__main__':
    clf = class_fun()
    dest = os.path.join('movieclassifier', 'pkl_objects')
    #创建目录，存储序列化后的python对象
    if not os.path.exists(dest):
        os.makedirs(dest)
    pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
    pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)