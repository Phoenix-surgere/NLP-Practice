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

#eda = df[:]
#EDA Result: All onions are sarcastic, all Huffpost are real
    #Relative balance of the two classes, 45% sarcasm
#eda['article_link_clean'] = eda['article_link'].apply(sliceurl)
#eda.drop(columns=['article_link', 'headline'],inplace=True)
#sarcastic = eda[eda['is_sarcastic'] == 1]
df.drop(columns=['article_link'], inplace=True)

sarfrac =df[df.is_sarcastic == 1].shape[0] / df.shape[0]
print('Percentage of Sarcastic comments: {}%'.format(round(sarfrac * 100)))

X,y = df[['headline']], df[['is_sarcastic']]
 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

seed = 666
oov_tok = "<OOV>"
max_len = 7       #For Padding 
vocab_size = 500  #For Tokenizer
embedding_dim = 16
train_fraction = 0.9
#train_size = int(df.shape[0] * train_fraction)

X_train, X_test, y_train, y_test = train_test_split(X,y, 
    train_size=train_fraction, random_state=seed)
                                                     
tokenizer = Tokenizer(num_words=vocab_size ,oov_token=oov_tok)
tokenizer.fit_on_texts(X_train) #Tokenizer on TRAIN SET
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X_train) 
padded = pad_sequences(sequences, maxlen=max_len)


#model = keras.models.Sequential([
#           keras.layers.Embedding() ])