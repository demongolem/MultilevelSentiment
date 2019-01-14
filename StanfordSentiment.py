'''
Created on Sep 27, 2018

@author: g.werner
'''

import Config
import json
import logging
from os import listdir
from os.path import isfile, join
from stanfordcorenlp import StanfordCoreNLP

def convert_scale(original):
    return original / 2.0 - 1.0

class StanfordSentiment(object):

    def __init__(self):
        self.props={'annotators': 'tokenize,ssplit,pos,parse,sentiment',
                    'pipelineLanguage':'en',
                    'outputFormat':'json',
                    'parse.model':'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
                    'sentiment.model':'C:\\Users\\g.werner\\Desktop\\GitRepositories\\k360.sentiment\\model\\stanford\\model-0000-70.74.ser.gz'
        }
        self.server_on = False

    def config(self, config_item):
        try:
            print('Using Stanford server: ' + config_item.STANFORD_SERVER + ':' + str(config_item.STANFORD_PORT))
            self.nlp = StanfordCoreNLP(config_item.STANFORD_SERVER, port=config_item.STANFORD_PORT, logging_level=logging.DEBUG)
            self.server_on = True
        except Exception as e:
            print('Error setting up server ' + str(e))

    # sentiment returns 0, 1, 2, 3, 4.  2 is neutral 4 is very positive and 0 is very negative
    # we will convert to the [-1,1] scale used by other annotators, so y = x/2.0 - 1.0 will be applied
    def evaluate_single_document(self, document, mode):
        if not self.server_on:
            return None
        annotations_text = self.nlp.annotate(document, properties=self.props)
        annotations = json.loads(annotations_text)
        sentences = annotations['sentences']
    
        sentence_sentiments = []
    
        mainSentiment = 0;
        longest = 0;
        
        print('Mode ' + mode)
        
        for sent in sentences:        
            sentiment = float(sent['sentimentValue'])
            print('Sent ' + str(sentiment))
            tokens = sent['tokens']
            last_token = tokens[-1]
            last_point = int(last_token['characterOffsetEnd'])
            if last_point > longest:
                mainSentiment = sentiment;
                longest = last_point;
            sentence_sentiments.append(convert_scale(sentiment))                
            
        print('Done')
            
        if mode == 'document':
            return [convert_scale(mainSentiment)]
        if mode == 'sentence':
            return sentence_sentiments
        return []

    def release_server(self):
        if hasattr('self', 'nlp'):
            self.nlp.close()

side_effect = []

def fetch_files(directory):
    global side_effect
    filelines = []
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    for onlyfile in onlyfiles:
        side_effect.append(onlyfile)
        with open(join(directory, onlyfile), 'r', encoding = "ISO-8859-1") as f:
            filelines.append(f.readlines())
    return filelines

if __name__ == '__main__':
    ss = StanfordSentiment()
    ss.config(Config.DevelopmentConfig, 'sentence')
    
    print('Fetching files')
    filelines = fetch_files('C:\\Users\\g.werner\\Desktop\\GitRepositories\\2018-08-16-annotation\\AnnotationTool\\training_sets\\Webhose1')
    
    print(len(filelines))
    
    end = 1
    
    for i in range(0, end):
        print(i)
        fileline = filelines[i]
        document = '\n'.join(fileline)
        print(ss.evaluate_single_document(document))

    ss.release_server()