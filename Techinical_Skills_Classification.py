import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import gensim
import re
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn import svm

train_df = pd.read_csv('Example_Technical_Skills.csv')
train_df['Label'] = 1
x = train_df['Technology Skills']
x_train_tokenized = [[w for w in sentence.split(" ") for sentence in x]]

print(x_train_tokenized[0])

test_df= pd.read_csv('Raw_Skills_Dataset.csv')
X = test_df['RAW DATA']

def clean_text(text):
    cleaned = re.sub("[a-zA-Z0-9]", " ",text)
    return cleaned.strip()

x_cleaned = [clean_text(t) for t in X]
x_test_tokenzied = [[w for w in sentence.split(" ") ] for sentence in x_cleaned]

model = gensim.models.Word2Vec(x_train_tokenized,
                               size=100,
                               window=5,
                               min_count=1,
                               workers=2,
                               sg=1)

class Sequencer():

    def __init__(self,
                 all_words,
                 max_words,
                 seq_len,
                 embedding_matrix
                 ):
        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_counts = {}

        for word in temp_vocab:
            count = len([0 for w in all_words if w == word])
            self.word_counts[word] = count
            counts = list(self.word_counts.values())
            indexes = list(range(len(counts)))

        count = 0
        while count +1 != len(counts):
            count = 0 
            for i in range(len(counts)-1):
                if counts[i] < counts[i+1]:
                    counts[i+1],counts[i] = counts[i],counts[i+1]
                    indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                else:
                    count += 1
        for ind indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])

    def textToVector(self, text):
        tokens = text.split()
        len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as E:
                    pass

        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
                vec.append(np.zeros(100,))
        return np.asarray(vec).flatten()


max_words = len(model.wv.vocab.items())
print(max_words)

sequencer = Sequencer(all_words = [token for seq in x_train_tokenized for token in seq],
             max_words = max_words,
             seq_len = 3,
             embedding_matrix = model.wv)

test_vec = sequencer.textToVector('Natural Language Processing')
print(test_vec)
print(test_vec.shape)

x_train_vecs = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in x_train_tokenized])
print(x_train_vecs.shape)
x_test_vecs = np.assarray([sequencer.textToVector(" ".join(seq)) for seq in x_test_tokenized])
print(x_test_vecs.shape)

pca_model = PCA(n_compnents=150)
pca_model.fit(x_train_vecs)
print(pca_model.explained_variance_ratio_)

x_train_comps = pca_model.transform(x_train_vecs)
x_test_comps = pca_model.transform(x_test_vecs)

clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.0000087)
clf.fit(x_train_comps)
print(clf.fit_status_))

predictions = clf.predict(x_test_comps)

x_normal_array = np.where(predictions == -1)
x_outlier_array = np.where(predictoins == 1)


