import gensim
import os
import random
import time
import datetime


def read_corpus(fname, tokens_only=False):
    with open(fname, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def get_d2v_model(train_x):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=50, workers=4)
    model.build_vocab(train_x)
    print("training the d2v_model...")
    st_time = time.time()
    model.train(train_x, total_examples=model.corpus_count, epochs=model.epochs)
    end_time = time.time()
    print("finish the training, time: %.2f" % (end_time - st_time))

    return model


def inc_train(model, train_x):
    model.build_vocab(train_x, update=True)
    model.train(train_x, total_examples=model.corpus_count, epochs=model.epochs)

    return model


def save_model(model):
    dt_str = datetime.datetime.now().strftime("%y%m%d")
    model.save(dt_str + ".d2v")


def query_sim(model, x):
    inferred_vector = model.infer_vector(x)
    sims = model.docvecs.most_similar([inferred_vector])
    return sims[0]


if __name__ == '__main__':
    train_data_url = os.path.join('processed_data', 'shuffle_train_data.csv')
    train_x = list(read_corpus(train_data_url))
    # print(train_x[:2])

    test_data_url = os.path.join('processed_data', 'shuffle_test_data.csv')
    test_x = list(read_corpus(test_data_url, tokens_only=True))
    print(test_x[:2])

    d2v_model = get_d2v_model(train_x)
    sim_index = query_sim(d2v_model, test_x[0])
    print(train_x[sim_index])


def get_ranks(d2v_model, train_x):
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_x)):
        inferred_vector = d2v_model.infer_vector(train_x[doc_id].words)
        sims = d2v_model.docvecs.most_similar([inferred_vector], topn=len(d2v_model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
        second_ranks.append(sims[1])

    print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_x[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_x[sims[index][0]].words)))

    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(train_x) - 1)
    # doc_id = 1371
    # Compare and print the most/median/least similar documents from the train corpus
    print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_x[doc_id].words)))
    sim_id = second_ranks[doc_id]
    print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_x[sim_id[0]].words)))

    # Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_x) - 1)
    inferred_vector = d2v_model.infer_vector(test_x[doc_id])
    sims = d2v_model.docvecs.most_similar([inferred_vector], topn=len(d2v_model.docvecs))

    # Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_x[doc_id])))

    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % d2v_model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_x[sims[index][0]].words)))


'''
def get_datasest():
    with open("out/wangyi_title_cut.txt", 'r') as cf:
        docs = cf.readlines()
        print(len(docs))

    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)

    return x_train

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train,min_count=1, window = 3, size = size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model/model_dm_wangyi')

    return model_dm

def test():
    model_dm = Doc2Vec.load("model/model_dm_wangyi")
    test_text = ['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print inferred_vector_dm
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)


    return sims
'''
