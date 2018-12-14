import pickle
import re
import os
from chapter9.movieclassifier.vectorizer import vect
import numpy as np

if __name__ == '__main__':
    clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

    label = {0: 'negative', 1: 'positive'}
    example = ['I\'m  happy']
    X = vect.transform(example)
    print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))