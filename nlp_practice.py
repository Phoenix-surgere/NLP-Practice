# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:58:04 2019

@author: black

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from pandas import DataFrame as DF
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer as WNL
lemmatizer = WNL()
string = 'I am a bay, but I am not an old fuck. Who said I am a wonderfully FUCK?!'
tokens = word_tokenize(string)

tokens = Counter(word_tokenize(string))
tokens = DF([tokens]); tokens = tokens.T
tokens.columns = ['frequencies']
tokens.sort_values(by='frequencies').plot(kind='bar')
plt.show()
tokens = [token.lower() for token in tokens]
from gensim.corpora.dictionary import Dictionary
tokens = [t for t in tokens if t not in stopwords.words('english')]
#dict_tokens = Dictionary(tokens)
"""

#NLTK Book Scripts go here
#from nltk import *
from nltk.book import *