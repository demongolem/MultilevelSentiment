#!/usr/bin/env python
# encoding: utf-8
'''
Created on Sep 18, 2018

@author: g.werner
'''
'''
Created on Sep 18, 2018

@author: g.werner
'''
import csv
from pattern.text.en import ngrams

reader = csv.reader(open('dict.csv', 'r'))
pharma_dict = dict((rows[0],rows[1]) for rows in reader)

def evaluate_single_document(document):
    # Pre-processing the extracted text using ngrams function from the pattern package   
    final_text1 = ngrams(document, n=1, punctuation=".,;:!?()[]{}`''\"@#$^&*+-|=~_", continuous=False)
       
    # Checking if any of the words in the news article text matches with the words in the Pharma dictionary(pos/neg)
    new_dict = {}
       
    for x in final_text1:
        if x[0] in pharma_dict:
            new_dict[x[0]] = pharma_dict[x[0]] 
           
    positive_list = [] ; negative_list = [];
    for key, value in new_dict.items():
        if value == 'positive': positive_list.append(key)
        if value == 'negative': negative_list.append(key)
                     
    # Compute the positive score, the negative score for each news articles
    positive_score = len(positive_list) ; negative_score = len(negative_list);

    return (positive_score, negative_score)