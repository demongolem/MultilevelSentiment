'''
Created on Sep 27, 2018

@author: g.werner
'''

import Config
import json
import logging
import os
from os import listdir
from os.path import isfile, join
import shutil
from stanfordcorenlp import StanfordCoreNLP
import subprocess
import sys
import time
import urllib.request
import zipfile

def convert_scale(original):
    return original / 2.0 - 1.0

class StanfordSentiment(object):

    def __init__(self):
        self.props={'annotators': 'tokenize,ssplit,pos,parse,sentiment',
                    'pipelineLanguage':'en',
                    'outputFormat':'json',
                    'parse.model':'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
                    'sentiment.model': os.path.realpath(__file__) + '/../model/stanford/model-0000-70.74.ser.gz'
        }
        self.server_on = False

    def config(self, config_item):
        try:
            print('Using Stanford server: ' + config_item.STANFORD_SERVER + ':' + str(config_item.STANFORD_PORT))
            self.nlp = StanfordCoreNLP(config_item.STANFORD_SERVER, port=config_item.STANFORD_PORT, logging_level=logging.DEBUG, max_retries=5)
            self.server_on = True
        except Exception as e:
            print('Error setting up server ' + str(e))
            print('We will try establish a server here')
            location = config_item.STANFORD_LOCATION
            if os.path.isdir(location):
                subprocess.Popen(['java','-mx4g','-cp',location + '\\*','edu.stanford.nlp.pipeline.StanfordCoreNLPServer','-port','9000','-timeout','15000'])
                # maybe a better way to accomplish this
                time.sleep(5)
                self.nlp = StanfordCoreNLP(config_item.STANFORD_SERVER, port=config_item.STANFORD_PORT, logging_level=logging.DEBUG, max_retries=5)
                self.server_on = True
            else:
                print("Can't find Stanford CoreNLP.   Downloading.....")                                
                
                url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip'
                file_name = 'stanford-corenlp-full-2018-02-27.zip'

                with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)                    
                    
                zip_ref = zipfile.ZipFile(file_name, 'r')
                zip_ref.extractall(location)
                zip_ref.close()
                source = os.path.join(location, 'stanford-corenlp-full-2018-02-27')
                    
                files = os.listdir(source)
                for f in files:
                    shutil.move(os.path.join(source, f), location)

                print("Downloading SR Parser model.....")  

                sr_parser_url = "https://nlp.stanford.edu/software/stanford-parser-full-2014-10-31.zip"
                sr_parser_file_name = 'stanford-parser-full-2014-10-31.zip'
                
                with urllib.request.urlopen(sr_parser_url) as response, open(sr_parser_file_name, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)    
                
                with zipfile.ZipFile(sr_parser_file_name, 'r') as zip_ref:
                        zip_ref.extractall(location)
    
                source = os.path.join(location, 'stanford-parser-full-2014-10-31')
                files = os.listdir(source)

                for f in files:
                    if f == 'stanford-parser.jar':
                        shutil.move(os.path.join(source, f), location)
                
                subprocess.Popen(['java','-mx4g','-cp',location + '\\*','edu.stanford.nlp.pipeline.StanfordCoreNLPServer','-port','9000','-timeout','15000'])
                # maybe a better way to accomplish this
                time.sleep(5)
                self.nlp = StanfordCoreNLP(config_item.STANFORD_SERVER, port=config_item.STANFORD_PORT, logging_level=logging.DEBUG, max_retries=5)
                self.server_on = True

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

    if len(sys.argv) < 3:
        print('Usage: python StanfordSentiment <test_files> <mode>')
        exit(1)

    file_directory = sys.argv[1]
    mode = sys.argv[2]
    
    ss = StanfordSentiment()
    ss.config(Config.DevelopmentConfig)
    
    print('Fetching files')
    filelines = fetch_files(file_directory)
    
    print('Found ' + str(len(filelines)))
    
    end = len(filelines)
    
    for i in range(0, end):
        print(i)
        fileline = filelines[i]
        document = '\n'.join(fileline)
        print(ss.evaluate_single_document(document, mode))

    ss.release_server()