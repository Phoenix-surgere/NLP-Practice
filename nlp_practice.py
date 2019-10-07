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

text1.concordance('monstrous') #Search for word, needs specific format to work
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
