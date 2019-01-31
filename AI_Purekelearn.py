import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
import os
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.test.utils import datapath

import os
import time
import gc
import re

from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D,GRU,Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.metrics import binary_accuracy



from sklearn.svm import SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics

from tqdm import tqdm
tqdm.pandas()


embed_size = 300 # how big is each word vector
max_features = 15000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 1500 # max number of words in a question to use


X = pd.read_csv("train_values.csv")["doc_text"]
y = pd.read_csv("train_labels.csv")
y =y.drop(y.columns[0],axis=1)
test = pd.read_csv("test_values.csv")["doc_text"]

print("Data import Completed")

#X_beta = X.iloc[:500]
#X_beta = tokenizer.texts_to_sequences(X_beta)
#X_beta = pad_sequences(X_beta, maxlen=maxlen)
#y_beta = y.iloc[:500,:]

tokenizer = Tokenizer(num_words=max_features,filters= '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n\r123456789 ', lower=True,split=' ')
tokenizer.fit_on_texts(X)

print("Tokenizing Completed")

X = tokenizer.texts_to_sequences(X)
pred_X = tokenizer.texts_to_sequences(test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

word_list = []
for word in word_index: 
    word_list.append(word)
    
X = pad_sequences(X, maxlen=maxlen)
pred_X = pad_sequences(pred_X,maxlen=maxlen)

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.1, random_state=311)

word_index = tokenizer.word_index
max_features = len(word_index)+1
def load_glove(word_index):
    EMBEDDING_FILE = 'glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix 

embedding_matrix_1 = load_glove(word_index)



model = Sequential()
model.add(Embedding(max_features, embed_size,weights=[embedding_matrix_1],input_length =maxlen,))
model.add(Bidirectional(CuDNNGRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(29,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

usualCallback = EarlyStopping()
overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience = 0)



model.fit(train_X, train_y, epochs=3, batch_size=300,validation_data=(test_X,test_y),callbacks=[overfitCallback])
    

pred_noemb_val_y = model.predict([X_beta], batch_size=1024, verbose=5)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(y_beta, (pred_noemb_val_y>thresh).astype(int),average='micro')))


y_pred = model.predict(pred_X,verbose=1,batch_size=1000)



def ToAnswer():
    pd.DataFrame(y_pred).to_csv("y_pred.csv",index=False)
    
y_pred_D = pd.DataFrame(y_pred)

y_pred_D[y_pred_D>=0.5]=1
y_pred_D[y_pred_D<0.5]=0

y_pred_D = y_pred_D.astype(int)

sub = pd.read_csv("submission_format.csv")

sub = pd.concat([sub["row_id"],y_pred_D],axis=1)
sub.columns = ['row_id', 'information_and_communication_technologies', 'governance',
       'urban_development', 'law_and_development', 'public_sector_development',
       'agriculture', 'communities_and_human_settlements',
       'health_and_nutrition_and_population', 'culture_and_development',
       'environment', 'social_protections_and_labor', 'industry',
       'macroeconomics_and_economic_growth',
       'international_economics_and_trade', 'conflict_and_development',
       'finance_and_financial_sector_development',
       'science_and_technology_development', 'rural_development',
       'poverty_reduction', 'private_sector_development', 'informatics',
       'energy', 'social_development', 'water_resources', 'education',
       'transport', 'water_supply_and_sanitation', 'gender',
       'infrastructure_economics_and_finance']

sub.to_csv("lstm_128_64_5.5.csv",index=False)


