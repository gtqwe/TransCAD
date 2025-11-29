import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from nltk import word_tokenize

data = pd.DataFrame(pd.read_excel('Contracts.xlsx'))
data['opcode'] = data['opcode'].str.replace('\n', ' ')

cw = lambda x: list(word_tokenize(x))
data['opcode'] = data['opcode'].apply(cw)

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['opcode'])
data['opcode'] = tokenizer.texts_to_sequences(data['opcode'])

words = tokenizer.word_index
print(len(words)+1)

avg_len = list(map(len, data['opcode']))
print(np.mean(avg_len))

maxlen = 3072
print("Pad sequences (samples x time)")
data['opcode'] = list(sequence.pad_sequences(data['opcode'], maxlen=maxlen, padding='post'))

from sklearn.utils import shuffle
data = shuffle(data, random_state=0)

X = np.array(list(data['opcode']))
y = np.array(list(data['label']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=0)

from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=0)
X_train, y_train = adasyn.fit_resample(X_train, y_train)
X_test, y_test = adasyn.fit_resample(X_test, y_test)

y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

length = 512
X_train1 = X_train[:, 0:length]
X_train2 = X_train[:, length:2 * length]
X_train3 = X_train[:, 2 * length:3 * length]
X_train4 = X_train[:, 3 * length:4 * length]
X_train5 = X_train[:, 4 * length:5 * length]
X_train6 = X_train[:, 5 * length:6 * length]

X_test1 = X_test[:, 0:length]
X_test2 = X_test[:, length:2 * length]
X_test3 = X_test[:, 2 * length:3 * length]
X_test4 = X_test[:, 3 * length:4 * length]
X_test5 = X_test[:, 4 * length:5 * length]
X_test6 = X_test[:, 5 * length:6 * length]

def positional_embedding(maxlen, model_size):
    PE = np.zeros((maxlen, model_size))
    for i in range(maxlen):
        for j in range(model_size):
            if j % 2 == 0:
                PE[i, j] = np.sin(i / 10000 ** (j / model_size))
            else:
                PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_size))
    PE = tf.constant(PE, dtype=tf.float32)
    return PE

class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, model_size, input_length):
        super(PositionalEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.input_length = input_length

        self.embedding = Embedding(vocab_size, model_size)
        self.pos_embedding = positional_embedding(input_length, model_size)

    def call(self, x):
        # input embedding + positional embedding
        x = self.embedding(x) + self.pos_embedding
        return x

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        
        self.attention = MultiHeadAttention(num_heads=num_heads,
                                            key_dim=embed_dim)
        
        self.dense_proj = Sequential([Dense(dense_dim, activation='relu'),
                                      Dense(embed_dim),])
        
        self.layernorm_1 = LayerNormalization()
        
        self.layernorm_2 = LayerNormalization()

    def call(self, inputs):
        
        attention_output = self.attention(inputs, inputs)
        
        proj_input = self.layernorm_1(inputs + attention_output)
       
        proj_output = self.dense_proj(proj_input)
        
        return self.layernorm_2(proj_input + proj_output)

print('Build model...')
inputs = [Input(shape=(length,), dtype='float32') for _ in range(6)]

branch_outputs = []
transformer_reps = []

for i in range(6):
    embed = PositionalEmbedding(len(words) + 1, 128, input_length=length)(inputs[i])
    
    if i == 0:
        x = embed
    else:
        x = concatenate(branch_outputs + [embed])
        x = Dense(128)(x)
    
    trans = TransformerEncoder(embed_dim=128, dense_dim=32, num_heads=2)(x)
    transformer_reps.append(trans)
    
    if i < 5:
        x = Dropout(0.2)(trans)
        out = Convolution1D(32, 3, padding='same', strides=1, activation='relu')(x)
        branch_outputs.append(out)

fusion = multiply(transformer_reps)

flat = Flatten()(fusion)
drop = Dropout(0.2)(flat)
output = Dense(3, activation='softmax')(drop)

model = Model(inputs, output)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([X_train1, X_train2, X_train3, X_train4, X_train5, X_train6], y_train,
          batch_size=32, epochs=10,
          validation_data=([X_test1, X_test2, X_test3, X_test4, X_test5, X_test6], y_test))

classes = model.predict([X_test1, X_test2, X_test3, X_test4, X_test5, X_test6])

