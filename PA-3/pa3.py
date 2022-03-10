
import argparse
import re
import time
from collections import Counter
from math import log

import numpy as np
import pandas as pd
import requests
from nltk.corpus import stopwords
from pattern.en import lemma

# variables
DATA_PATH_BASE = "./data/"
OUTPUT_PATH_BASE = "./output/"
DATA_URL = "https://ceiba.ntu.edu.tw/course/88ca22/content/training.txt"


def parser():
    parser = argparse.ArgumentParser(description="pa3 homework")
    parser.add_argument("--time", action='store_true',
                        help="show time in each steps")
    return parser.parse_args()


def timer(start_time=None, title=""):
    """
    start_time = timer(None)
    timer(start_time)
    """
    args = parser()
    if args.time is True:
        if not start_time:
            start_time = time.process_time()
            return start_time
        elif start_time:

            print('\n' + title + " \t" +
                  str(round(time.process_time() - start_time, 5)), " seconds")


def pattern_stopiteration_workaround():
    try:
        print(lemma('gave'))
    except Exception:
        pass


def download_data(DATA_URL):
    data = requests.get(DATA_URL)
    class_data = [x.split(' ')
                  for x in data.text.split('\n')]

    out_filename = "class.csv"
    with open(out_filename, 'w+') as file:
        for i in class_data:
            file.write((",".join(i[:-1])+"\n"))


def read_training_data():
    train_class_dict = {}  # doc id in each class, likelihood
    docs_token = {}  # token freq in each doc, likelihood
    train_term = []  # all the terms, likelihood

    # read raw training data
    start_time = timer(None)
    with open("class.csv", "r") as f:
        for line in f.read().splitlines():
            data = list(map(int, line.split(",")))
            c = data[0]  # class num
            docs = data[1:]  # doc id in the class

            train_class_dict[c] = docs

            for i in docs:
                with open("data/" + str(i) + ".txt") as f:
                    doc = f.read()
                    token = tokenize(doc)
                    train_term.extend(token)

                    freq = Counter(token)  # calc doc freq
                    docs_token[i] = freq

        train_term = sorted(list(set(train_term)))
    return train_class_dict, docs_token, train_term


def tokenize(corpus) -> list:

    txt = corpus.lower()
    txt = re.sub(r'[^\w\s]', ' ', txt)  # clear all non-word
    txt = re.sub('[0-9]', '', txt)
    txt = re.sub('_', '', txt)
    word = txt.split()
    new = []
    stop = set(stopwords.words('english'))
    for w in word:
        if w not in stop:
            new.append(lemma(w))  # using pattern.en to do lemmatize
    return new


class MultinomialNB():
    def __init__(self):
        self.prior = []
        self.condprob = {}

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        classes, counts = np.unique(
            y, return_counts=True)  # classes = [ 1 to 13 ]
        classes -= 1

        docs, terms = X.shape  # extract vocab, count dos

        # count prior
        self.prior = counts / sum(counts)
        class_term_freq = [np.zeros(terms) for _ in classes]

        for idx, row in enumerate(X):  # for each doc
            class_term_freq[y[idx]-1] += row
        self.condprob = [np.zeros(terms) for _ in classes]

        for c in classes:  # count doc in class
            class_term_count = np.count_nonzero(class_term_freq[c], axis=0)
            self.condprob[c] = (class_term_freq[c] + 1) / \
                (class_term_count + 500)

        self.condprob = np.log(self.condprob)

    def predict(self, X):
        X = np.array(X)
        docs, terms = X.shape

        expand_prior = np.tile(np.log(self.prior), (docs, 1))
        post = (X != np.zeros(terms)) @ self.condprob.T
        res = np.argmax(expand_prior + post, axis=1) + 1
        return res


class LikelihoodRatio():
    def __init__(self, method="SUM", total_features=500):
        self.method = method
        self.total_features = total_features
        self.likelihood = {}
        self.likelihood_ratio = {}
        self.sum_likelihood_ratio = {}

    def _generate_term_matrix(self):

        start_time = timer(None)

        for word in self.all_term:
            self.likelihood[word] = np.zeros((self.class_count, 2))
            for c in range(1, 14):

                for i in self.class_dict[c]:
                    doc = self.token_freq[i].keys()

                    if word in doc:
                        self.likelihood[word][c-1][0] += 1
                    else:
                        self.likelihood[word][c-1][1] += 1
        timer(start_time, "generate matrix")

    def _calc_likelihood_ratio(self):

        # calculate avg likelihood ratio
        start_time = timer(None)

        for word in self.all_term:
            term_stats = []
            stat = self.likelihood[word]
            N = np.sum(stat)
            sum_p, sum_a = np.sum(stat, axis=0)

            for c in range(0, self.class_count):
                n11, n10 = stat[c]
                n01, n00 = sum_p - n11, sum_a - n10

                pt = (n11 + n01) / N
                p1 = ((n11) / (n11 + n10))
                p2 = ((n01) / (n01 + n00))

                h1 = ((pt**n11) * ((1-pt)**n10) * (pt**n01) * ((1-pt)**n00))
                h2 = ((p1**n11) * ((1-p1)**n10) * (p2**n01) * ((1-p2)**n00))

                ratio = -2 * (log(h1) - log(h2))

                term_stats.append(ratio)

            self.likelihood_ratio[word] = term_stats
            self.sum_likelihood_ratio[word] = sum(term_stats)/self.class_count
        timer(start_time, "calc likelihook ratio")

    def _select_top_k_features(self):
        start_time = timer(None)

        feature_per_class = int(
            (self.total_features - (self.total_features % self.class_count)) / self.class_count)
        total_features = []
        for c in range(0, self.class_count):
            class_vocab = {}
            for term in self.all_term:
                class_vocab[term] = self.likelihood_ratio[term][c]

            max_features = list(dict(sorted(class_vocab.items(),
                                            key=lambda item: item[1], reverse=True)).keys())[:feature_per_class]

            total_features.extend(max_features)
        timer(start_time, "select topk features")

        return list(set(total_features))

    def _select_top_avg_features(self):
        return sorted(list(dict(sorted(
            self.sum_likelihood_ratio.items(), key=lambda item: item[1], reverse=True)[:self.total_features]).keys()))

    def fit(self, all_term, class_dict, token_freq):
        self.all_term = all_term
        self.class_dict = class_dict
        self.class_count = len(class_dict.keys())
        self.token_freq = token_freq

        self._generate_term_matrix()
        self._calc_likelihood_ratio()

    def transform(self):
        if self.method == "SUM":
            return self._select_top_avg_features()
        if self.method == "TOPK":
            return self._select_top_k_features()


if __name__ == "__main__":

    try:
        open("class.csv", "r")
    except Exception:
        download_data(DATA_URL)

    start_time = timer(None)
    pattern_stopiteration_workaround()  # lemma workaround
    timer(start_time, "pattern workaround")

    # read training data
    start_time = timer(None)
    train_class_dict, docs_token, train_term = read_training_data()
    timer(start_time, "read data")

    # feature selection
    start_time = timer(None)
    LIKE = LikelihoodRatio("TOPK", 500)
    LIKE.fit(train_term, train_class_dict, docs_token)
    selected_features = LIKE.transform()
    timer(start_time, title="feature selection")

    # prepare training data
    start_time = timer(None)
    trainX = []
    trainy = []
    for i in range(1, 14):
        for docID in train_class_dict[i]:
            doc_x = []
            for vocab in selected_features:
                if vocab in docs_token[docID].keys():
                    doc_x.append(docs_token[docID][vocab])
                else:
                    doc_x.append(0)
            trainX.append(doc_x)
            trainy.append(i)
    X = np.array(trainX)
    y = np.array(trainy)
    timer(start_time, "prepare train data")

    # prepare testing data
    start_time = timer(None)
    testX = []
    test_id = pd.read_csv("hw3_sam.csv")["Id"]
    for i in test_id:
        with open("data/" + str(i) + ".txt") as f:
            token = tokenize(f.read())
            freq = Counter(token)
            doc_x = []
            for vocab in selected_features:
                if vocab in freq.keys():
                    doc_x.append(freq[vocab])
                else:
                    doc_x.append(0)
            testX.append(doc_x)
    X_test = np.array(testX)
    timer(start_time, "prepare testing data")

    # train NB model
    start_time = timer(None)
    clf = MultinomialNB()
    clf.fit(X, y)
    timer(start_time, "NB train")

    # test NB model
    start_time = timer(None)
    res = clf.predict(X_test)
    timer(start_time, "NB test")

    out = pd.read_csv("hw3_sam.csv")
    out["Value"] = pd.Series(res)
    # out.to_csv("result.csv")
    out.set_index("Id").to_csv("result.csv")
