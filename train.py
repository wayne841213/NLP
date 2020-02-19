# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:15:55 2020

@author: wayne.kuo
"""

import pandas as pd
#from nltk.corpus import stopwords 




#prestore it as txt to make inread data prettier
#change sub label to sub_label
dataset = pd.read_csv(r'tweet label2.txt',  engine = "python", index_col=False, skiprows = 0, 
        encoding ="ISO-8859-1", na_values = '-', delimiter =',', skipinitialspace=True, quotechar='"')
    
dataset.head()


#
import os

#import sys

from collections import namedtuple

import numpy as np

import pandas as pd

from keras_xlnet.backend import keras

from keras_bert.layers import Extract

from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

from keras_radam import RAdam

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config)) 

### 预训练模型的路径

pretrained_path  = "./xlnet_cased_L-12_H-768_A-12"

EPOCH = 10

BATCH_SIZE = 1

SEQ_LEN = 256

PretrainedPaths = namedtuple('PretrainedPaths', ['config', 'model', 'vocab'])

config_path = os.path.join(pretrained_path, 'xlnet_config.json')

model_path = os.path.join(pretrained_path, 'xlnet_model.ckpt')

vocab_path = os.path.join(pretrained_path, 'spiece.model')

paths = PretrainedPaths(config_path, model_path, vocab_path)

tokenizer = Tokenizer(paths.vocab)

#
# Read data

class DataSequence(keras.utils.Sequence):

    def __init__(self, x, y):

        self.x = x

        self.y = y

    def __len__(self):

        return (len(self.y) + BATCH_SIZE - 1) // BATCH_SIZE

    def __getitem__(self, index):

        s = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)

        return [item[s] for item in self.x], self.y[s]

def generate_sequence(df):

    tokens, classes = [], []
#    a=0
    for _, row in df.iterrows():

        ###这里笔者将数据进行拼接  类型+问题1+问题2

        text, cls = row["fulltext"], row['label']

        try:
            Label = int(cls)
            encoded = tokenizer.encode(text)[:SEQ_LEN - 1]
        except:
#            print(text)
            continue
#        a = max(a,len(encoded))
        if len(encoded)==255:
            print(text)
        encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]

        tokens.append(encoded)
        classes.append(Label)
#    print(a)
    tokens, classes = np.array(tokens), np.array(classes)

    segments = np.zeros_like(tokens)

    segments[:, -1] = 1

    lengths = np.zeros_like(tokens[:, :1])

    return DataSequence([tokens, segments, lengths], classes)

### 读取数据，然后将数据

data_path = 'tweet label2.txt'

data = dataset

test = data.sample(200)

train = data.loc[list(set(data.index)-set(test.index))]


### 生成训练集和测试集

train_g = generate_sequence(train)

test_g = generate_sequence(test)

#%% Load pretrained model

model = load_trained_model_from_checkpoint(

    config_path=paths.config,

    checkpoint_path=paths.model,

    batch_size=BATCH_SIZE,

    memory_len=0,

    target_len=SEQ_LEN,

    in_train_phase=False,

    attention_type=ATTENTION_TYPE_BI,

)

#### 加载预训练权重

# Build classification model

last = model.output

extract = Extract(index=-1, name='Extract')(last)

dense = keras.layers.Dense(units=768, name='Dense')(extract)

norm = keras.layers.BatchNormalization(name='Normal')(dense)

output = keras.layers.Dense(units=11, activation='softmax', name='Softmax')(norm)

model = keras.models.Model(inputs=model.inputs, outputs=output)

model.summary()

# 定义优化器，loss和metrics

model.compile(

    optimizer=RAdam(learning_rate=1e-4),

    loss='sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy'],

)

### 定义callback函数，只保留val_sparse_categorical_accuracy 得分最高的模型

from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("./model/best_xlnet.h5", monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True,

                            mode='max')

#模型训练

model.fit_generator(

    generator=train_g,

    validation_data=test_g,

    epochs=EPOCH,

    callbacks=[checkpoint],

)