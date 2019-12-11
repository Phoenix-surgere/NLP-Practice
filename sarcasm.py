# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:52:53 2019

@author: black
1:Is sarcastic, 0:ain't
"""
import pandas as pd
import re
from nltk.corpus import stopwords 
df1= pd.read_json("news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json",
                 lines=True)

df2= pd.read_json("news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json",
                 lines=True)

df = pd.concat([df1, df2], axis=0, ignore_index=True)

#Remove stopwords via nltk
stop_words = set(stopwords.words('english'))
df['headline']= df['headline'].apply(lambda x: ' '.join([word for word in x.split() if 
           word not in (stop_words)]))

def sliceurl(link):
    result = re.split(r"://", re.split("\.com/", link)[0])[1]
    if 'com' in result:
        result = re.sub(r'\.co.+', "",result)
    return result

eda = df[:]
#EDA Result: All onions are sarcastic, all Huffpost are real
#Relative balance of the two classes, 45% sarcasm, so accuracy should work as a metric and no special technigues needed for imbalance
eda['article_link_clean'] = eda['article_link'].apply(sliceurl)
eda.drop(columns=['article_link', 'headline'],inplace=True)
sarcastic = eda[eda['is_sarcastic'] == 1]
df.drop(columns=['article_link'], inplace=True)

sarfrac =df[df.is_sarcastic == 1].shape[0] / df.shape[0]
print('Percentage of Sarcastic comments: {}%'.format(round(sarfrac * 100)))

X,y = df[['headline']], df[['is_sarcastic']]
 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import keras

seed = 666
oov_tok = "<OOV>"
max_len = 7       #For Padding 
vocab_size = 500  #For Tokenizer
embedding_dim = 16
train_fraction = 0.9

X_train, X_test, y_train, y_test = train_test_split(X,y, 
    train_size=train_fraction, random_state=seed)

X_train = X_train['headline'].tolist()
X_test = X_test['headline'].tolist()

tokenizer = Tokenizer(num_words=vocab_size ,oov_token=oov_tok)
tokenizer.fit_on_texts(X_train) #Tokenizer on TRAIN SET
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(X_train) 
train_padded = pad_sequences(training_sequences, maxlen=max_len)

testing_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(testing_sequences, maxlen=max_len)


model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
    ])
    
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
history = model.fit(train_padded, y_train, 
          validation_data=(test_padded, y_test), epochs=50)
plot_loss_metric(history)
