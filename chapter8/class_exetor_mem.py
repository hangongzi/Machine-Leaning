import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind

stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]>', '',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv)
        for line in csv:
            text, line = line[:-3], int(line[-2])
            yield text, line

def get_minibatch(doc_stream, size):
    docs, y =[], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

#量化添加数据进行模型拟合
def class_fun():
    vect = HashingVectorizer(decode_error='ignore',
                                 n_features=2**21,
                                 preprocessor=None,
                                 tokenizer=tokenizer)
    clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
    doc_stream = stream_docs('movie_data.csv')
    pbar = pyprind.ProgBar(45)
    classes = np.array([0, 1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream,size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
        pbar.update()
    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' % clf.score(X_test, y_test))
    clf = clf.partial_fit(X_test, y_test)

if __name__ == '__main__':
    class_fun()
