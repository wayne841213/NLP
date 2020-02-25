# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:57:11 2020

@author: wayne.kuo
"""

import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
# For reproducibility
np.random.seed(1237)
# Source file directory
# downlaod from http://qwone.com/~jason/20Newsgroups/

dataset = pd.read_csv(r'tweet label_lin_20190902.csv',  engine = "python", index_col=False, skiprows = 0, 
        encoding ="ISO-8859-1", na_values = '-', delimiter =',', skipinitialspace=True, quotechar='"')
    
dataset.head()


filter = dataset['fulltext']==dataset['fulltext']
data = dataset[filter] # 篩選 data frame
filter =  dataset['label']==dataset['label']
data = data[filter] # 篩選 data frame

data['label'] = data['label'].map(int)
#data['sub_label'] = data['sub_label'].map(int)


for index,row in data.iterrows():
    if row['sub_label']!=row['sub_label']:
        data.loc[index,'sub_label'] = row['label']
    else:
        try:
            data.loc[index,'sub_label'] = int (row['sub_label'])
        except:
            data.loc[index,'sub_label'] = row['label']

        
data['sub_label'] = data['sub_label'].map(int)


#data = data.where(pd.notnull(data), None)
data['label'] = data[['label', 'sub_label']].apply(tuple, axis=1)



from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit_transform(data['label'])
mlb.classes_




contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }
from nltk.corpus import stopwords 
import codecs
import unidecode
import re
import spacy
nlp = spacy.load('en')

STOPWORDS = set(stopwords.words('english'))

def spacy_cleaner(text):
    #lower case
    text = text.lower()
    #unicode_escape for extra "\" before unicode character, then unidecode
    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)
    #there are apostrophe and singlequote people use for contraction. any apostrophe) is changed to single quote
    apostrophe_handled = re.sub("’", "'", decoded)
    #Contraction check
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        #Filtering punctuation, white space, numbers, URL using Spacy methods while keeping the text content of hashtag intact
        #Removed @mention
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):            
            pass
        else:
            #Pronouns are kept as they are since Spacy lemmatizer transforms every pronoun to "-PRON-"
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                #Lemmatize: lemmatized each token using Spacy method '.lemma_'.
                #Special character removal
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                #Single syllable token removal
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    """joined = ' '.join(PorterStemmer().stem(spell(word)) for word in final_tokens if word not in STOPWORDS) """ 
    # delete stopwors from text,  finding the base word, auto correct
    joined = ' '.join(final_tokens)
    #it is a simple spell correction, if the same character is repeated more than two times, it shortens the repetition to two
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


data['fulltext'] = data['fulltext'].astype(str).apply(spacy_cleaner)

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('history.jpg')



num_labels = len(mlb.classes_)
vocab_size = 300
batch_size = 100
num_epochs = 10000
max_features = 2000
maxlen = vocab_size
embedding_dims = 50
kernel_size = 3
hidden_dims = 250
filters = 250

#20 news groups
# lets take 80% data as training and remaining 20% for test.
train_size = int(len(data) * .9)

        
train_posts = data['fulltext'][:train_size]
train_tags = data['label'][:train_size]
train_files_names = data['status_id'][:train_size]

test_posts = data['fulltext'][train_size:]
test_tags = data['label'][train_size:]
test_files_names = data['status_id'][train_size:]

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_posts)

x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')



encoder = MultiLabelBinarizer()
encoder.fit(train_tags)

y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


data_label_count =data.label.value_counts()
x_train.shape, y_train.shape





#%%let us build a basic model





from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D

"""
model = Sequential()
#model.add(Dense(512, input_shape=(vocab_size,)))
#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(Dense(num_labels))
#model.add(Activation('softmax'))
#model.summary()



#model.add(Embedding(max_features,
#                    embedding_dims,
#                    input_length=maxlen))
#model.add(Dropout(0.2))
#
#model.add(Conv1D(filters,
#                 kernel_size,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
#
#model.add(GlobalMaxPooling1D())

model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))


model.add(Dense(num_labels))
model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=2,
                    validation_split=0.2)

plot_history(history)

score, acc = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=2)

print('Test accuracy:', acc)
"""
#%% load the whole embedding into memory (很久)

from numpy import zeros, asarray

embeddings_index = dict()
f = open('glove.twitter.27B.200d.txt',encoding="utf-8")

for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
    
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
        
#%%
#
#another approach using GRU model, takes longer time
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(train_posts) 

# create a weight matrix for words in training docs

# define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1 

embedding_matrix = zeros((vocab_size, 200))

for word, i in tokenizer_obj.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


# pad sequences
max_length = max([len(s.split()) for s in train_posts])



X_train_tokens =  tokenizer_obj.texts_to_sequences(train_posts)
X_test_tokens = tokenizer_obj.texts_to_sequences(test_posts)


X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')


encoder = MultiLabelBinarizer()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

def balance(x):
    bg=x.max()
    y={}
    for n,v in enumerate(x):
        y[n] = round(bg/v)
    return y

class_weights = balance( y_train.sum(axis=0))
class_weights[5] = 30

from keras.models import Sequential
from keras.layers import Dense,BatchNormalization , Embedding, LSTM, CuDNNGRU, GRU, Flatten, Bidirectional
from keras.layers.embeddings import Embedding
from keras import regularizers

EMBEDDING_DIM = 32
num_epochs = 1000

print('Build model...')




model = Sequential()
#model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))

model.add(Embedding(vocab_size, 200, weights=[embedding_matrix],
                    input_length=max_length, trainable=False))

model.add(Bidirectional(GRU(units=16, dropout=0.2, recurrent_dropout=0.2,
              recurrent_regularizer = regularizers.l2(0.01),
#              kernel_regularizer=regularizers.l2(0.01),
#              activity_regularizer=regularizers.l1(0.01),
              return_sequences=True
              )))
model.add(Bidirectional(GRU(units=16, dropout=0.2, recurrent_dropout=0.2,
              recurrent_regularizer = regularizers.l2(0.01),
#              kernel_regularizer=regularizers.l2(0.01),
#              activity_regularizer=regularizers.l1(0.01),
              )))


model.add(BatchNormalization())

model.add(Dense(num_labels, activation='softmax'))


from keras.utils import multi_gpu_model
# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
model = multi_gpu_model(model, gpus=2)


model.load_weights("BidirectionalGRU.hdf5")

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



print('Summary of the built model...')
print(model.summary())


from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("BidirectionalGRU.hdf5",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             )

history = model.fit(X_train_pad, y_train,
                    batch_size=50,
                    epochs=num_epochs,
                    verbose=1,
#                    validation_split=0.1,
                    validation_data=(X_test_pad, y_test),
                    shuffle=True,
                    class_weight=class_weights,
                    callbacks=[checkpoint],
                    )

#score, acc = model.evaluate(X_test_pad, y_test,
#                       batch_size=batch_size,
#                       verbose=1)
#
#print('Test accuracy:', acc)
plot_history(history)





#%%

text_labels = encoder.classes_

#for i in range(10):
#    prediction = model.predict(np.array([x_test[i]]))
#    predicted_label = text_labels[np.argmax(prediction[0])]
#    #print(test_files_names.iloc[i])
#    print('Actual label:' + test_tags.iloc[i])
#    print("Predicted label: " + predicted_label)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


x_test = X_test_pad

y_test.sum(axis=0)

y_pred = model.predict(x_test);
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
#cnf_matrix = confusion_matrix( asarray(data['label']),  asarray(data['sub_label']))

# Plot normalized confusion matrix
fig = plt.figure()
fig.set_size_inches(14, 12, forward=True)
#fig.align_labels()

label_names=['work','industry','politic','life and family','charity'
,'flaunt wealth'
,'sport event or hobbies','public activity','personal development of the CEO'
,'current events and social issues','others']

# fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plot_confusion_matrix(cnf_matrix, classes=np.asarray(label_names), normalize=True,
                      title='Normalized confusion matrix')

fig.savefig("txt_classification-smote" + str(num_epochs) + ".png", pad_inches=5.0)