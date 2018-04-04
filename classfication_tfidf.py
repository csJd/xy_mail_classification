import os
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import xgboost
import pickle



def load_data(url):
    """ get all words use specified text file

    :param url:
    :return:
    """
    dic = dict()
    lines = list()
    y = list()
    with open(url, encoding='utf-8') as txt:
        for line in txt.readlines():
            lines.append(line[2:])
            y.append(int(line[0]))
            words = line[2:].split()
            for word in words:
                if word in dic:
                    dic[word] += 1
                else:
                    dic[word] = 1

    # return sorted(dic.items(), key=lambda x: x[1], reverse=True)
    return lines, y


def load_test(test_url):
    lines, y = load_data(test_url)


def get_cntv_train(train_url):
    """
    :param train_url:
    :return:
    """
    lines, _ = load_data(train_url)
    cntv = CountVectorizer()
    return cntv.fit_transform(lines)


def skl_tf_idf(train_url, test_url):
    """

    :param url:
    :return:
    """
    lines, y = load_data(train_url)
    train_vectorizer = CountVectorizer()
    train_count = train_vectorizer.fit_transform(lines)
    y = np.array(y)

    test_lines, testy = load_data(test_url)
    # use vocabulary from train data
    test_vectorizer = CountVectorizer(vocabulary=train_vectorizer.vocabulary_)
    test_count = test_vectorizer.fit_transform(test_lines)
    testy = np.array(testy)

    # word = vectorizer.get_feature_names()
    # print(word)  # see count
    # print(X.toarray)  # count vector

    train_transformer = TfidfTransformer()
    X = train_transformer.fit_transform(train_count, y)
    # print(X.shape, y.shape)

    test_transformer = TfidfTransformer()
    testX = test_transformer.fit_transform(test_count, testy)
    # print(testX.shape, testy.shape)

    return X, y, testX, testy


def predict(clf, X, y):
    return clf.predict(X)
    # print(precision_recall_fscore_support(y, predy))
    # print(accuracy_score(y, predy))


def clf_stacking(clfs, X, y):
    y_pred = np.zeros(y.shape)
    for clf in clfs:
        y_pred += predict(clf, X, y)
    y_pred = y_pred > 1.5
    print(accuracy_score(y, y_pred))



def main():
    """ Test the module
    :return:
    """

    train_url = os.path.join("processed_data", "shuffle_train_data.csv")  # url to train file
    # test_url = os.path.join("processed_data", "shuffle_test_data.csv")  # url to test file
    test_url = os.path.join("processed_data", "test.csv")  # url to test file
    # dic = get_dict(txt_url)
    # with open('dic.txt', 'w', encoding='utf-8') as dic_file:
    #    dic_file.write(str(len(dic)) + str(dic))
    X, y, testX, testy = skl_tf_idf(train_url, test_url)
    clf_svm = svm.SVC()
    clf_knn = KNeighborsClassifier()
    clf_rf = RandomForestClassifier()
    clf_gbdt = GradientBoostingClassifier()
    clf_xgb = xgboost.XGBClassifier()

    clf = clf_svm.fit(X, y)
    # predict(clf, testX, testy)


    clf = clf_knn.fit(X, y)
    # predict(clf, testX, testy)

    clfs = list()
    clf_rf.fit(X, y)
    # predict(clf_rf, testX, testy)
    joblib.dump(clf_rf, 'rf.pkl')
    clfs.append(clf_rf)

    clf_gbdt.fit(X, y)
    # predict(clf_gbdt, testX, testy)
    joblib.dump(clf_gbdt, 'gbdt.pkl')
    clfs.append(clf_gbdt)

    clf_xgb.fit(X, y)
    # predict(clf, testX, testy)
    joblib.dump(clf_xgb, 'xgb.pkl')
    clfs.append(clf_xgb)

    clf_stacking(clfs, testX, testy)


if __name__ == '__main__':
    main()




