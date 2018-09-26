'''
Author: Zeping Yu
2018.9.26
'''

import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K
from keras import optimizers,initializers,regularizers,constraints
from keras.engine.topology import Layer, InputSpec
from keras.models import Sequential, Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Embedding, Activation, Input, Lambda, Reshape, CuDNNGRU, CuDNNLSTM, GlobalMaxPooling1D, concatenate, BatchNormalization
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, Merge, LSTM, GRU, Bidirectional, TimeDistributed
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,TensorBoard
from keras.optimizers import SGD, Adam, Adadelta

df2 = pd.read_csv("yelp_2013.csv")
df2['text']=df2['text'].astype(str)

Y = df2.stars.values-1
Y = to_categorical(Y,num_classes=5)
X = df2.text.values

MAX_NB_WORDS = 30000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2
NUM_FILTERS = 50
MAX_LEN = 200
BATCH_SIZE = 100
EPOCHS = 30
WINDOW_SIZE = 15

indices = np.arange(X.shape[0])
np.random.seed(2018)
np.random.shuffle(indices)
X=X[indices]
Y=Y[indices]

nb_validation_samples_val = int(VALIDATION_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples_val]
y_train = Y[:-nb_validation_samples_val]
x_val =  X[-nb_validation_samples_val:]
y_val =  Y[-nb_validation_samples_val:]

tokenizer1 = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer1.fit_on_texts(df2.text)

x_train_word_ids = tokenizer1.texts_to_sequences(x_train)
x_val_word_ids = tokenizer1.texts_to_sequences(x_val)
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=MAX_LEN)
x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=MAX_LEN)

embeddings_index = {}
f = open('glove.6B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((MAX_NB_WORDS + 1, EMBEDDING_DIM))
for word, i in tokenizer1.word_index.items():
    if i<MAX_NB_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be random.
            embedding_matrix[i] = embedding_vector
        
class DRNN(Layer):
    def __init__(self, window_size, hidden_size, maxlen, **kwargs):
        super(DRNN, self).__init__(**kwargs)
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.maxlen = maxlen

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.hidden_size

    def call(self, inputs): #assume inputs is [2,5,4], window_size is 3, hidden_size is 6
        pad_input = tf.pad(inputs,[[0,0],[self.window_size-1,0],[0,0]]) #pad_input is [2,7,4]
        rnn_inputs = []
        for i in range(self.maxlen):
            rnn_inputs.append(K.expand_dims(tf.slice(pad_input,[0,i,0],[-1,self.window_size,-1]),1)) 
        rnn_input_tensor = tf.concat(rnn_inputs,1) #rnn_input_tensor is [2,5,3,4]
        timegru = TimeDistributed(CuDNNGRU(self.hidden_size))(rnn_input_tensor) #timegru is [2,5,6]
        return timegru

from keras import regularizers
embedding_layer = Embedding(MAX_NB_WORDS + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LEN)
#                            embeddings_regularizer=regularizers.l2(0.0001))

from keras.layers import concatenate
main_input = Input(shape=(MAX_LEN,), dtype='float64')
embed = embedding_layer(main_input)
drnn = DRNN(WINDOW_SIZE, NUM_FILTERS, MAX_LEN)(embed)
#bn = BatchNormalization()(drnn)
'''
gru = CuDNNGRU(NUM_FILTERS, return_sequences=True)(embed) #gru implementation
cnn = Convolution1D(NUM_FILTERS, WINDOW_SIZE)(embed) #cnn implementation
'''
pool = GlobalMaxPooling1D()(drnn) #change drnn into gru or cnn when comparing
main_output = Dense(5, activation='softmax')(pool)
model = Model(inputs = main_input, outputs = main_output)
print model.summary()

adadelta = Adadelta()
adam = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['acc'])
model.fit(x_train_padded_seqs, y_train, 
          validation_data = (x_val_padded_seqs, y_val),
          nb_epoch = EPOCHS, 
          batch_size = BATCH_SIZE,
          verbose = 1)
