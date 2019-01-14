#!/usr/bin/env python
# encoding: utf-8
'''
Created on Sep 20, 2018

@author: g.werner
'''

import re 
from textblob import TextBlob
  
def clean_tweet(tweet): 
    ''' 
    Utility function to clean tweet text by removing links, special characters 
    using simple regex statements. 
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])  |(\w+:\/\/\S+)", " ", tweet).split()) 
  
def get_tweet_sentiment(tweet): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(clean_tweet(tweet)) 
    return analysis.sentiment.polarity
  
def evaluate_single_document(document):
    answer = get_tweet_sentiment(document)
    return answer