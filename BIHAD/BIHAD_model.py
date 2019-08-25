import sys
import keras.models
from keras.layers.core import *
from keras.models import *
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers import Input,concatenate
from keras.layers import Conv1D, MaxPooling1D,TimeDistributed
from keras.layers import Dense,Flatten
from keras.layers import Bidirectional,Embedding,BatchNormalization,AveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def dense_block(input_tensor, channels, name=None):
    bn1 = BatchNormalization()(input_tensor)
    relu = Activation('relu')(bn1)
    conv = Conv1D(channels, 3, padding='same')(relu)
    return conv
def dense_b_block(input_tensor, channels, name=None):
    bn1 = BatchNormalization()(input_tensor)
    relu = Activation('relu')(bn1)
    conv = Conv1D(4 * channels, 5, padding='same')(relu)
    bn2 = BatchNormalization()(conv)
    relu2 = Activation('relu')(bn2)
    conv2 = Conv1D(channels, 3, padding='same')(relu2)
    return conv2
def transition_layer(input_tensor, k, name=None):

    conv = Conv1D(k, 5 , padding='same')(input_tensor)

    pool = AveragePooling1D(pool_size=2 )(conv)

    return pool

TEXT_DATA_DIR2 ='../Texture/'
Dictionary_Path2='../texture_embedding_model.txt'
TEXT_DATA_DIR3 ='../char/'
Dictionary_Path3='../char_embedding_model.txt'
TEXT_DATA_DIR4 ='../word/'
Dictionary_Path4='../word_embedding_model.txt'

save_path='best_model.h5'
EMBEDDING_DIM = 300
rate=0.2
epoch=50
MAX_SEQUENCE_LENGTH2 = 500
MAX_SEQUENCE_LENGTH3 = 500
MAX_SEQUENCE_LENGTH4 = 500
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #use GPU with ID=0


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)

labels_index = {}  # dictionary mapping label name to numeric id
labels2 = []  # list of label ids
i=1

texts2=[]
for name in sorted(os.listdir(TEXT_DATA_DIR2)):
    path = os.path.join(TEXT_DATA_DIR2, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            t = f.read()
            t = t.split()
            texts2.append(t)
            f.close()
            labels2.append(label_id)

MAX_NB_WORDS = 0
tokenizer2 = Tokenizer(num_words=MAX_NB_WORDS,filters="",oov_token="unk")
tokenizer2.fit_on_texts(texts2)
sequences2 = tokenizer2.texts_to_sequences(texts2)
word_index2 = tokenizer2.word_index
print(word_index2)
print('Found %s unique tokens.' % len(word_index2))
data2 = pad_sequences(sequences2,maxlen=MAX_SEQUENCE_LENGTH2)#, maxlen=MAX_SEQUENCE_LENGTH
labels2 = to_categorical(np.asarray(labels2))
print('Shape of data tensor:', data2.shape)
print('Shape of label tensor:', labels2.shape)

indices = np.arange(data2.shape[0])
np.random.shuffle(indices)
data2 = data2[indices]
labels2 = labels2[indices]

nb_validation_samples = int(rate * data2.shape[0])
label2=[]
for la in labels2:
    if la[0]==0:
        label2.append(la[0])
    else:
        label2.append(la[0])

labels_index = {}  # dictionary mapping label name to numeric id
labels3 = []  # list of label ids
i=1
texts3=[]
for name in sorted(os.listdir(TEXT_DATA_DIR3)):
    path = os.path.join(TEXT_DATA_DIR3, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            t = f.read()
            t = t.split()
            texts3.append(t)
            f.close()
            labels3.append(label_id)

tokenizer3 = Tokenizer(num_words=MAX_NB_WORDS,filters="",oov_token="unk")
tokenizer3.fit_on_texts(texts3)
sequences3 = tokenizer3.texts_to_sequences(texts3)
word_index3 = tokenizer3.word_index
print(word_index3)
print('Found %s unique tokens.' % len(word_index3))
data3 = pad_sequences(sequences3,maxlen=MAX_SEQUENCE_LENGTH3)#, maxlen=MAX_SEQUENCE_LENGTH
labels3 = to_categorical(np.asarray(labels3))
print('Shape of data tensor:', data3.shape)
print('Shape of label tensor:', labels3.shape)


data3 = data3[indices]
labels3 = labels3[indices]
nb_validation_samples = int(rate * data3.shape[0])
label3=[]
for la in labels3:
    if la[0]==0:
        label3.append(la[0])
    else:
        label3.append(la[0])

labels_index = {}  # dictionary mapping label name to numeric id
labels4 = []  # list of label ids
i=1
texts4=[]
for name in sorted(os.listdir(TEXT_DATA_DIR4)):
    path = os.path.join(TEXT_DATA_DIR4, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            t = f.read()
            t = t.split()
            texts4.append(t)
            f.close()
            labels4.append(label_id)

tokenizer4 = Tokenizer(num_words=MAX_NB_WORDS,filters="",oov_token="unk")
tokenizer4.fit_on_texts(texts4)
sequences4 = tokenizer4.texts_to_sequences(texts4)
word_index4 = tokenizer4.word_index
print(word_index4)
print('Found %s unique tokens.' % len(word_index4))
data4 = pad_sequences(sequences4,maxlen=MAX_SEQUENCE_LENGTH4)#, maxlen=MAX_SEQUENCE_LENGTH
labels4 = to_categorical(np.asarray(labels4))
print('Shape of data tensor:', data4.shape)
print('Shape of label tensor:', labels4.shape)

data4 = data4[indices]
labels4 = labels4[indices]
nb_validation_samples = int(rate * data4.shape[0])
label4=[]
for la in labels4:
    if la[0]==0:
        label4.append(la[0])
    else:
        label4.append(la[0])

##########################################################
x_train2 = data2[:-nb_validation_samples]
y_train2 = label2[:-nb_validation_samples]
x_val2 = data2[-nb_validation_samples:]
y_val2 = label2[-nb_validation_samples:]
embeddings_index2 = {}
f = open( Dictionary_Path2,'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index2[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index2))
embedding_matrix2 = np.zeros((len(word_index2) + 1, EMBEDDING_DIM))
print(word_index2)
for word, i in word_index2.items():
    embedding_vector2 = embeddings_index2.get(word)
    if embedding_vector2 is not None and embedding_vector2.shape[0]==EMBEDDING_DIM:
        # words not found in embedding index will be all-zeros.
        embedding_matrix2[i] = embedding_vector2
print("embedding_matrix",embedding_matrix2)
##########################################################
x_train3 = data3[:-nb_validation_samples]
y_train3 = label3[:-nb_validation_samples]
x_val3 = data3[-nb_validation_samples:]
y_val3 = label3[-nb_validation_samples:]
embeddings_index3 = {}
f = open( Dictionary_Path3,'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index3[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index3))
embedding_matrix3 = np.zeros((len(word_index3) + 1, EMBEDDING_DIM))
print(word_index3)
for word, i in word_index3.items():
    embedding_vector3 = embeddings_index3.get(word)
    if embedding_vector3 is not None and embedding_vector3.shape[0]==EMBEDDING_DIM:
        # words not found in embedding index will be all-zeros.
        embedding_matrix3[i] = embedding_vector3
print("embedding_matrix3",embedding_matrix3)
##########################################################
x_train4 = data4[:-nb_validation_samples]
y_train4 = label4[:-nb_validation_samples]
x_val4 = data4[-nb_validation_samples:]
y_val4 = label4[-nb_validation_samples:]
embeddings_index4 = {}
f = open( Dictionary_Path4,'r',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index4[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index4))
embedding_matrix4 = np.zeros((len(word_index4) + 1, EMBEDDING_DIM))
print(word_index4)
for word, i in word_index4.items():
    embedding_vector4 = embeddings_index4.get(word)
    if embedding_vector4 is not None and embedding_vector4.shape[0]==EMBEDDING_DIM:
        # words not found in embedding index will be all-zeros.
        embedding_matrix4[i] = embedding_vector4
print("embedding_matrix4",embedding_matrix4)

#################################################################################
#  model

from IndRNN import IndRNN
sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH2,), dtype='int32')
te_ind = Embedding(len(word_index2) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix2],
                            input_length=MAX_SEQUENCE_LENGTH2,
                            trainable=True)(sequence_input2)

te_ind = Bidirectional(IndRNN(64, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))(te_ind)
texture=Dense(400,activation='relu')(te_ind)

sequence_input3 = Input(shape=(MAX_SEQUENCE_LENGTH3,), dtype='int32')
ch_ind = Embedding(len(word_index3) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix3],
                            input_length=MAX_SEQUENCE_LENGTH3,
                            trainable=False)(sequence_input3)
char=Dense(400,activation='relu')(ch_ind)
sequence_input4 = Input(shape=(MAX_SEQUENCE_LENGTH4,), dtype='int32')
wo_ind = Embedding(len(word_index4) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix4],
                            input_length=MAX_SEQUENCE_LENGTH4,
                            trainable=False)(sequence_input4)
words=Dense(400,activation='relu')(wo_ind)
k = 12
output = concatenate([texture,char,words])
conv1d_out= Conv1D(kernel_size=1, filters=k, padding='same',activation='tanh', strides=1)(output)
x=MaxPooling1D(pool_size=3)(conv1d_out)
b1_1 = dense_b_block(x, k)
b1_1_conc = keras.layers.concatenate([x, b1_1], axis=1)
b1_2_conc = keras.layers.concatenate([x, b1_1,b1_1_conc], axis=1)
b1_3_conc = keras.layers.concatenate([x, b1_1,b1_1_conc,b1_2_conc], axis=1)
b1_4_conc = keras.layers.concatenate([x, b1_1,b1_1_conc,b1_2_conc,b1_3_conc], axis=1)
pool1 = transition_layer(b1_4_conc, k)
b2_1 = dense_b_block(pool1, k)
b2_1_conc = keras.layers.concatenate([pool1, b2_1], axis=1)
b2_2_conc = keras.layers.concatenate([x, b2_1,b2_1_conc], axis=1)
b2_3_conc = keras.layers.concatenate([x, b2_1,b2_1_conc,b2_2_conc], axis=1)
b2_4_conc = keras.layers.concatenate([x, b2_1,b2_1_conc,b2_2_conc,b2_3_conc], axis=1)
pool2 = transition_layer(b2_4_conc, k)
b3_1 = dense_b_block(pool2, k)
b3_1_conc = keras.layers.concatenate([pool2, b3_1], axis=1)
b3_2_conc = keras.layers.concatenate([x, b3_1,b3_1_conc], axis=1)
pool3 = transition_layer(b3_2_conc, k)
from attention import Attention_layer
output=Attention_layer()(pool3)
preds = Dense(1, activation='sigmoid')(output)
print("training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
model = Model(inputs=[ sequence_input2,sequence_input3,sequence_input4], outputs=[preds])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['acc'])
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath=save_path,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only='True',
                             save_weights_only='True',
                             mode='max',
                        period=1)
import time
fit_start = time.clock()
history=model.fit([x_train2,x_train3,x_train4], y_train2,
          batch_size=128,
          epochs=epoch,
          verbose=2,
          shuffle =True,
          validation_data=([x_val2,x_val3,x_val4], y_val2),
            callbacks = [checkpoint])
model.load_weights(save_path)
scores =  model.evaluate([x_val2,x_val3,x_val4], y_val2,verbose=0)
