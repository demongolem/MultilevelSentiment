#!/usr/bin/env python
# encoding: utf-8
'''
Created on Sep 19, 2018

@author: g.werner
'''

import Config
import json
from lib_model.bidirectional_lstm import LSTM
import logging
import nltk
from nltk import Tree
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from os import listdir
from os.path import isfile, join
from queue import Queue
from stanfordcorenlp import StanfordCoreNLP

nltk.download('punkt')

# for testing only please!  Use the server created in Entry => StanfordSentiment please for deployment usage
def getCoreNlpInstance(config_item):
    # don't need sentiment, however the stanford annotator does need it
    props={'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,coref,sentiment',
            'pipelineLanguage':'en',
            'outputFormat':'json',
            'parse.model':'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
            'sentiment.model': os.path.realpath(__file__) + '/../model/stanford/model-0000-70.74.ser.gz'
    }
    # we do not provide the same level of recovery as in StanfordSentiment.  Please manually start your server first
    return StanfordCoreNLP(config_item.STANFORD_SERVER, port=config_item.STANFORD_PORT, logging_level=logging.DEBUG, max_retries=5,  memory='8g')

def convert_scale(positive):
    return 2 * positive - 1

def flatten(input_list):
    return [val for sublist in input_list for val in sublist]

def tree_to_str(tree):
    return ' '.join([w for w in tree.leaves()])

def get_rep_mention(coreference):
    for reference in coreference:
        if reference['isRepresentativeMention'] == True:
            pos = (reference['startIndex'], reference['headIndex'])
            text = reference['text']
            return text, pos

def get_subtrees(tree):
    """ Return chunked sentences """
    
    subtrees = []
    queue = Queue()
    queue.put(tree)
    
    while not queue.empty():
        node = queue.get()
        
        for child in node:
            if isinstance(child, Tree):
                queue.put(child)
                
        if node.label() == "S":
            # if childs are (respectively) 'NP' and 'VP'
            # convert subtree to string, else keep looking

            # TODO: MAKE SURE NP IS A PERSON
            child_labels = [child.label() for child in node]

            if "NP" in child_labels and "VP" in child_labels:
                sentence = tree_to_str(node)
                for child in node:
                    if child.label() == "NP":
                        # look for NNP
                        subchild_labels = [subchild.label() for subchild in child]
                        if "NNP" in subchild_labels:
                            noun = ""
                            for subchild in child:
                                if subchild.label() == "NNP":
                                    noun = ' '.join([noun, subchild.leaves()[0]])

                            subtrees.append((noun, sentence))
    return subtrees

class CharLSTMSentiment(object):

    def __init__(self):
        self.network = LSTM()
        self.network.build()
        self.server_on = False 

    def config(self, config, nlp):
        self.nlp = nlp
        self.server_on = True

    def init_dict(self):
        local_dict = {}
        for k, _ in self.contexts:
            if not k in local_dict:
                local_dict[k] = None
        self.entities = local_dict

    def evaluate_single_document(self, document, mode):
        if mode == 'document':
            document = document[0:1000]
            p = self.network.predict_sentences([document])
            positive = p[0][0][0]
            return [convert_scale(positive)]
        elif mode == 'sentence':
            return self.evaluate_sentences(sent_tokenize(document))
        elif mode == 'entity':
            return self.get_entity_sentiment(document)
        else:
            return ['UNKNOWN MODE']

    #sentence sentiment function
    def evaluate_sentences(self, sentences):
        scores = []
        p = self.network.predict_sentences(sentences)
        for i in range(0, len(sentences)):
            positive = p[0][i][0]
            scores.append(convert_scale(positive))
        return scores

    # the following in this class all have to do with entity sentiment
    # we need to make sure it is serializable to json (i.e. beware of float32)
    def get_entity_sentiment(self, document):
        """ Create a dict of every entities with their associated sentiment """
        print('Parsing Document...')
        self.parse_doc(document)
        print('Done Parsing Document!')
        self.init_dict()
        #sentences = [sentence.encode('utf-8') for _, sentence in self.contexts]
        sentences = [sentence for _, sentence in self.contexts]
        print('Predicting!')
        predictions = self.network.predict_sentences(sentences)

        for i, c in enumerate(self.contexts):
            key = c[0]
            if self.entities[key] != None:
                self.entities[key] += (predictions[0][i][0] - predictions[0][i][1])
                self.entities[key] /= 2
            else:
                self.entities[key] = (predictions[0][i][0] - predictions[0][i][1])
    
        for e in self.entities.keys():
            # conversion for json purposes
            self.entities[e] = str(self.entities[e])
            print('Entity: %s -- sentiment: %s' % (e, self.entities[e]))
            
        return self.entities

    def parse_doc(self, document):
        """ Extract relevant entities in a document """
        print('Tokenizing sentences...')
        # why are we mixing nlp pipelines here?
        #nltk
        sentences = sent_tokenize(document)
        print('Done Sentence Tokenize!')
        # Context of all named entities
        ne_context = []
        for sentence in sentences:
            # change pronouns to their respective nouns
            print('Anaphora resolution for sentence: %s' % sentence)
            (output, modified_sentence) = self.coreference_resolution(sentence)
            tree = self.parse_sentence(output, modified_sentence)
            print('Done Anaphora Resolution!')
    
            # get context for each noun
            print('Named Entity Clustering:')
            context = get_subtrees(tree)
            for n, s in context:
                print('%s' % s)
            ne_context.append(context)
        self.contexts = flatten(ne_context)
    
    def coreference_resolution(self, sentence):
        # coreference resolution
        # corenlp
        print('Starting document annotation for ' + sentence)
        output_string = self.nlp.annotate(sentence)
        print('Done document annotation')
        output = json.loads(output_string)
        coreferences = output['corefs']
        entity_keys = coreferences.keys()

        tokens = word_tokenize(sentence)
        
        for k in entity_keys:
            # skip non PERSON NP
            if coreferences[k][0]['gender'] == 'MALE' or coreferences[k][0]['gender'] == 'FEMALE':
                rep_mention, pos = get_rep_mention(coreferences[k])
                for reference in coreferences[k]:
                    if not reference['isRepresentativeMention']:
                        start, end = reference['startIndex'] - 1, reference['headIndex'] - 1
                        if start == end:
                            tokens[start] = rep_mention
                        else:
                            tokens[start] = rep_mention
                            del tokens[start + 1: end]
    
        sentence = ' '.join(tokens)
        print('Ending coref function')
        return (output, sentence.encode('utf-8'))
    
    def parse_sentence(self, output, sentence):
        """ sentence --> named-entity chunked tree """
        try:
            return Tree.fromstring(output['sentences'][0]['parse'])
        except TypeError as e:
            import pdb; pdb.set_trace()

side_effect = []

def fetch_files(directory):
    global side_effect
    filelines = []
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    for onlyfile in onlyfiles:
        side_effect.append(onlyfile)
        with open(join(directory, onlyfile), 'r', encoding="utf-8") as f:
            filelines.append(f.readlines())
    return filelines

if __name__ == '__main__':
    cls = CharLSTMSentiment()
    config_item = Config.DevelopmentConfig
    cls.config(config_item, getCoreNlpInstance(config_item))
    document = 'Bob talked with the great ruler John yesterday.  John mentioned how horrible Tesla is.  The nefarious Bob agreed.'

    print('Fetching files')
    filelines = fetch_files('input/test')
    
    print(len(filelines))

    limit_files_to = 10
    
    for i in range(0, len(filelines)):
        if i == limit_files_to:
            break
        print(i)
        fileline = filelines[i]
        document = '\n'.join(fileline)
        result = cls.evaluate_single_document(document, 'entity')
        print(result)  
