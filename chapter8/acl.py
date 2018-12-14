import pyprind
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

def pro_data():
    pbar = pyprind.ProgBar(50000)
    labels = {'pos':1, 'neg':0}
    df = pd.DataFrame()
    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = 'D:\\workspace\\Machine-Learning\\chapter8\\aclImdb_v1\\aclImdb\\%s\\%s'%(s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'rb') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
    df.columns = ['review', 'sentiment']
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('movie_data.csv', index=False)

def creat_word_bags():
    count = CountVectorizer()
    docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining and the weather is sweet'
    ])
    bag = count.fit_transform(docs)
    # print(count.vocabulary_)
    # print(bag.toarray())
    tfidf = TfidfTransformer()
    np.set_printoptions(precision=2)
    #print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

def prepeocessor(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+',' ',text.lower()) + ''.join(emoticons).replace('-','')
    return text

def tokenizer(text):
    return text.split()

#词干提取
def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

#获取NLTK库提供的停用词
def get_stop_words():
    nltk.download('stopwords')
    stop = stopwords.words('english')
    w_list = [w for w in tokenizer_porter(' a runner likes runing and runs a lot')[-10:] if w not in stop]
    print(w_list)

def class_movie_words():
    nltk.download('stopwords')
    stop = stopwords.words('english')
    df = pd.read_csv('movie_data.csv')
    df = tokenizer_porter(df)
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    param_grid = [{'vect__ngram_range': [(1,1)],
                   'vect__stop_words':[stop, None],
                   'vect__tokenizer':[tokenizer, tokenizer_porter],
                   'clf__penalty':['l1', 'l2'],
                   'clf_C':[1.0, 10.0, 100.0]},
                  {'vect__ngram_range':[(1, 1)],
                   'vect__stop_words':[stop, None],
                   'vect__tokenizer':[tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty':['l1', 'l2'],
                   'clf__C':[1.0, 10.0, 100.0]}
                  ]
    lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)
    print('Best parameter set: %s'% gs_lr_tfidf.best_params_)

def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv)
        for line in csv:
            text, label = line[: -3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

if __name__ == '__main__':
    pass
