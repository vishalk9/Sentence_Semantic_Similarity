import codecs
from nltk.tokenize import word_tokenize
import numpy as np
import os


def zero_pad(X, seq_len):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])


def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1  # plus the 0th word


def fit_in_vocabulary(X, voc_size):
    return [[w for w in x if w < voc_size] for x in X]


def load_sts(dsfile, glove, skip_unlabeled=True):
    """ load a dataset in the sts tsv format """
    s0 = []
    s1 = []
    labels = []
    with codecs.open(dsfile, encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            col= line.split('\t')
            label, s0x, s1x =col[4], col[5], col[6]
            if label == '':
                continue
            else:
                score_int = int(round(float(label)))
                y = [0] * 6
                y[score_int] = 1
                labels.append(np.array(y))
            for i, ss in enumerate([s0x, s1x]):
                words = word_tokenize(ss)
                index = []
                for word in words:
                    word = word.lower()
                    if word in glove.w:
                        index.append(glove.w[word])
                    else:
                        index.append(glove.w['UKNOW'])
                left = 100 - len(words)
                pad = [0]*left
                index.extend(pad)
                if i == 0:
                    s0.append(np.array(index))
                else:
                    s1.append(np.array(index))
            #s0.append(word_tokenize(s0x))
            #s1.append(word_tokenize(s1x))
    print len(s0)
    return (np.array(s0), np.array(s1), np.array(labels))


def load_embedded(glove, s0, s1, labels, ndim=0, s0pad=25, s1pad=60):
   

    if ndim == 1:
        # for averaging:
        e0 = np.array(glove.map_set(s0, ndim=1))
        e1 = np.array(glove.map_set(s1, ndim=1))
    else:
        
        e0 = glove.pad_set(glove.map_set(s0), s0pad)
        e1 = glove.pad_set(glove.map_set(s1), s1pad)
    return (e0, e1, s0, s1, labels)

def load_set(glove, file):
    
    s0, s1, labels = load_sts(file, glove)
    #s0, s1, labels = np.array(s0), np.array(s1), np.array(labels)
    print('(%s) Loaded dataset: %d' % (file, len(s0)))
    #e0, e1, s0, s1, labels = load_embedded(glove, s0, s1, labels)
    return ([s0, s1], labels)

def get_embedding():
    gfile_path = os.path.join("./glove.6B", "glove.6B.300d.txt")
    f = open(gfile_path, 'r')
    embeddings = {}
    for line in f:
        sp_value = line.split()
        word = sp_value[0]
        embedding = [float(value) for value in sp_value[1:]]
        embeddings[word] = embedding
    print "read word2vec finished!"
    f.close()
    return embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]