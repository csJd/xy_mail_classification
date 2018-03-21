import os
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost



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
    return dic, lines, y



def skl_tf_idf(url):
    """

    :param url:
    :return:
    """
    dic, lines, y = load_data(url)
    vectorizer = CountVectorizer()
    tf_train = vectorizer.fit_transform(lines)

    y = np.array(y)
    # word = vectorizer.get_feature_names()
    # print(word)  # see tf
    # print(X.toarray)  # tf vector

    transformer = TfidfTransformer()
    X = transformer.fit_transform(X, y)
    # print(X.shape)
    # print(y.shape)
    return X, y


def clfs(X, y):
    """

    :param X:
    :param y:
    :return:
    """


    clf_svm = svm.SVC()
    clf_svm.fit(X, y)

    clf_knn = NearestNeighbors()
    clf_knn.fit(X, y)

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X, y)

    clf_gbdt = GradientBoostingClassifier()
    clf_gbdt.fit(X, y)

    clf_xgb = xgboost.XGBClassifier()
    clf_xgb.fit(X, y)




def main():
    """ Test the module
    :return:
    """

    txt_url = os.path.join("processed_data", "shuffle_train_data.csv")  # url to csv file
    # dic = get_dict(txt_url)
    # with open('dic.txt', 'w', encoding='utf-8') as dic_file:
    #    dic_file.write(str(len(dic)) + str(dic))
    X, y = skl_tf_idf(txt_url)



if __name__ == '__main__':
    main()




