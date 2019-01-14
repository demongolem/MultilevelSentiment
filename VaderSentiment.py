#!/usr/bin/env python
# encoding: utf-8
'''
Created on Sep 19, 2018

@author: g.werner
'''
from langdetect import detect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import nltk
import os

nltk.download('punkt')
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

def load_documents():
    bm_data = []
    print('Getting data')    
    directory = os.fsencode('C:\\Users\\g.werner\\eclipse-workspace\\GenreDeciderPython\\input\\test')
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), mode='r', encoding="utf-8") as file:
            data = file.read().strip()
            if len(data) == 0:
                continue
            if detect(data) != 'en':
                continue            
            bm_data.append(data)
    return bm_data

def evaluate_single_document(document):
    ss = sid.polarity_scores(document)
    return ss['compound']

def main():
    documents = load_documents()

    pos = 0
    neu = 0
    neg = 0

    for document in documents:
        compound_value = evaluate_single_document(document)
        if compound_value <= -0.5:
            neg += 1
        elif compound_value >= 0.5:
            pos += 1        
        else:
            neu += 1
    print(str(pos) + ' pos ' + str(neu) + ' neu ' + str(neg) + ' neg')

if __name__ == '__main__':
    main()