'''
Author: Zeping Yu
2018.9.26
'''

import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Embedding, Input, CuDNNGRU, GlobalMaxPooling1D, BatchNormalization, TimeDistributed
from keras.layers import Convolution1D, Dropout, GRU

df2 = pd.read_csv("yelp_2013.csv") #The information about the test dataset could be found at https://github.com/zepingyu0512/srnn
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
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(x_train)
x_train_word_ids = tokenizer.texts_to_sequences(x_train)
x_val_word_ids = tokenizer.texts_to_sequences(x_val)
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
for word, i in tokenizer.word_index.items():
    if i<MAX_NB_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be random.
            embedding_matrix[i] = embedding_vector
        
class DRNN(Layer):
    '''
    Disconnected-rnn layer.
    Follows the work of Baoxin Wang. [http://aclweb.org/anthology/P18-1215]
    "Disconnected Recurrent Neural Networks for Text Categorization"
    by using recurrent units instead of convolution filters.
    # Input shape
        3D tensor with shape: `(batch_size, sequence_length, input_dim)`.
    # Output shape
        3D tensor with shape: `(batch_size, sequence_length, hidden_dim)`.
    How to use:
    Just put it on top of a 3D-tensor Layer (Embedding/others).
    The output dimensions are inferred based on the hidden dim of the RNN.
    Note: The layer has been tested with Keras 2.1.5, Python 2.7.14
    Example:
        model.add(Embedding(30000, 200, input_length=30))
        model.add(DRNN(50, 15))
        # next add a MaxPooling1D layer or whatever...
    '''
    def __init__(self, hidden_size, window_size, **kwargs):
        super(DRNN, self).__init__(**kwargs)
        self.window_size = window_size
        self.hidden_size = hidden_size

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.hidden_size

    def call(self, inputs): #assume inputs is [2,5,4], window_size is 3, hidden_size is 6
        pad_input = tf.pad(inputs,[[0,0],[self.window_size-1,0],[0,0]]) #pad_input is [2,7,4]
        drnn_inputs = []
        seqlen = inputs.shape[1]
        for i in range(seqlen):
            drnn_inputs.append(K.expand_dims(tf.slice(pad_input,[0,i,0],[-1,self.window_size,-1]),1)) 
        drnn_input_tensor = tf.concat(drnn_inputs,1) #rnn_input_tensor is [2,5,3,4]
        drnn_output = TimeDistributed(CuDNNGRU(self.hidden_size))(drnn_input_tensor) #drnn_output is [2,5,6]
        return drnn_output

embedding_layer = Embedding(MAX_NB_WORDS + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_LEN)
#                            embeddings_regularizer=regularizers.l2(0.0001))

main_input = Input(shape=(MAX_LEN,), dtype='float64')
embed = embedding_layer(main_input)
drnn = DRNN(NUM_FILTERS, WINDOW_SIZE)(embed)
#drnn = BatchNormalization()(drnn)
'''
gru = CuDNNGRU(NUM_FILTERS, return_sequences=True)(embed) #gru implementation
cnn = Convolution1D(NUM_FILTERS, WINDOW_SIZE)(embed) #cnn implementation
'''
pool = GlobalMaxPooling1D()(drnn) #change drnn into gru or cnn when comparing
main_output = Dense(5, activation='softmax')(pool)
model = Model(inputs = main_input, outputs = main_output)
print model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
model.fit(x_train_padded_seqs, y_train, 
          validation_data = (x_val_padded_seqs, y_val),
          nb_epoch = EPOCHS, 
          batch_size = BATCH_SIZE,
          verbose = 1)
