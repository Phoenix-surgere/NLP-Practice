# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:58:04 2019

@author: black
"""


#NLTK Book Scripts go here
#Chapter 1
from nltk.book import text3 as bible,  \           #to avoid star imports
text1 as moby_dick, text4 as inaugural_address, \
text2 as Jane_austen, text5 as chat

text1.concordance('monstrous') #Search for word, ONLY WORKS with * import! 
text2.similar('monstrous')     #similar words, differs per text!
text3.similar('God')

Jane_austen.common_contexts(["monstrous", 'very'])  #Common contexts for words

#Dispersion text tells us distribution of each word along the text's progression
inaugural_address.dispersion_plot(['citizens' , 'democracy', 'freedom', 'duties', 'America', 'God']) 
bible.dispersion_plot(['Jesus', 'God'])

chat.generate()      #generate text based on style of input
bible.generate()      #each time is different
inaugural_address.generate()  #according to the text!

texts = [bible, moby_dick, inaugural_address,
         Jane_austen, chat]

for text in texts:
    print('{}'.format(text), len(text))  #sample chosen texts' nominal sizes
    
def lexical_diversity(text):
    '''
       How many times a word appears on average on given text. 
       Higher means poorer diversity, lower better diversity
    '''   
    return(len(text) / len(set(text)))
    
def percent_word(text,word):
  ''' What percent of the text constitutes of the target- word'''
    count = text.count(word)
    percent = 100*count/ len(text)
    return percent

#print(percent_word(chat, 'lol'))
#list.append("item") -> append to end of list

' '.join(['Monty' ,'Python']) #join list to str
'Monty Python'.split()        #split str to list
import matplotlib.pyplot as plt
import pandas as pd
from nltk.probability import FreqDist
fdist_moby = FreqDist(moby_dick)  #frequency distribution
fdist_bible = FreqDist(bible)
fdist_chat = FreqDist(chat)

print(fdist_moby.most_common(10))
fdist_moby.plot(50, cumulative=True); plt.show()
print(fdist_bible.most_common(10))
print(fdist_chat.most_common(10))

print(len(fdist_moby.hapaxes()))  #words that occur once only
long_words = [word for word in moby_dick if len(word) >= 15] #long words 
long_words = sorted(set([w for w in chat if len(w) > 6 and 
              fdist_chat[w] > 7])) #long words w/ more conditions
#length of word and numer of occurences via FreqDist
from nltk import *
print(bigrams(['more', 'is' , 'said', 'than', 'done']))    

print('; '.join(chat.collocation_list())) #common bigrams

fmoby = FreqDist([len(w) for w in moby_dick]) #freq dist of lenghts of words!
print(fmoby.items())  #keys-values : length, apperances of words

#page 44 for more functions!
#page 45 for word comparison operators

#Examples of list comprehensions 
words_with_gnt = [w for w in moby_dick if 'gnt' in w]
words_with_initial_capital = [w for w in Jane_austen if w.istitle()]
words_only_digits = [w for w in chat if w.isdigit()]
words_lower_unique = len(set([w.lower() for w in chat if w.isalpha() ])) #true vocab
#HERE: TO SOLVE SOME CHAPTER 1 EXERCISES
