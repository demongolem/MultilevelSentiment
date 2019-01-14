#!/usr/bin/env python
# encoding: utf-8
'''
Created on Sep 19, 2018

@author: g.werner
'''

import Config
from lib_model.bidirectional_lstm import LSTM
import nltk
from nltk import Tree
from nltk.tokenize import sent_tokenize, word_tokenize
from os import listdir
from os.path import isfile, join
from pycorenlp import StanfordCoreNLP
from queue import Queue

nltk.download('punkt')

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

    def config(self, config):
        try:
            self.nlp = StanfordCoreNLP(config.STANFORD_SERVER + ':' + str(config.STANFORD_PORT))
            self.server_on = True
        except Exception as e:
            print('Stanford server could not be found')
            print(e)

    def init_dict(self):
        local_dict = {}
        for k, _ in self.contexts:
            if not k in local_dict:
                local_dict[k] = None
        self.entities = local_dict

    def parse_sentence(self, sentence):
        """ sentence --> named-entity chunked tree """
        try:
            output = self.nlp.annotate(sentence.decode('utf-8'), properties={'annotators':   'tokenize, ssplit, pos,'
                                                                        ' lemma, ner, parse',
                                                        'outputFormat': 'json'})
            # print_tree(output)
            return Tree.fromstring(output['sentences'][0]['parse'])
        except TypeError as e:
            import pdb; pdb.set_trace()

    def coreference_resolution(self, sentence):
        # coreference resolution
        output = self.nlp.annotate(sentence, properties={'annotators':   'coref',
                                                    'outputFormat': 'json'})
        tokens = word_tokenize(sentence)
        coreferences = output['corefs']
        entity_keys = coreferences.keys()
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
        return sentence.encode('utf-8')

    def parse_doc(self, document):
        """ Extract relevant entities in a document """
        print('Tokenizing sentences...')
        sentences = sent_tokenize(document)
        print('Done!')
        # Context of all named entities
        ne_context = []
        for sentence in sentences:
            # change pronouns to their respective nouns
            print('Anaphora resolution for sentence: %s' % sentence)
            tree = self.parse_sentence(self.coreference_resolution(sentence))
            print('Done!')
    
            # get context for each noun
            print('Named Entity Clustering:')
            context = get_subtrees(tree)
            for n, s in context:
                print('%s' % s)
            ne_context.append(context)
        self.contexts = flatten(ne_context)

    def get_entity_sentiment(self, document):
        """ Create a dict of every entities with their associated sentiment """
        print('Parsing Document...')
        self.parse_doc(document)
        print('Done!')
        self.init_dict()
        #sentences = [sentence.encode('utf-8') for _, sentence in self.contexts]
        sentences = [sentence for _, sentence in self.contexts]
        predictions = self.network.predict_sentences(sentences)

        for i, c in enumerate(self.contexts):
            key = c[0]
            if self.entities[key] != None:
                self.entities[key] += (predictions[0][i][0] - predictions[0][i][1])
                self.entities[key] /= 2
            else:
                self.entities[key] = (predictions[0][i][0] - predictions[0][i][1])
    
        for e in self.entities.keys():
            print('Entity: %s -- sentiment: %s' % (e, self.entities[e]))
            
        return self.entities

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

    def evaluate_sentences(self, sentences):
        scores = []
        p = self.network.predict_sentences(sentences)
        for i in range(0, len(sentences)):
            positive = p[0][i][0]
            scores.append(convert_scale(positive))
        return scores

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
    cls.config('document', Config.StagingConfig())
    document = 'Bob talked with the great ruler John yesterday.  John mentioned how horrible Tesla is.  The nefarious Bob agreed.'

    print('Fetching files')
    filelines = fetch_files('input/train')
    
    print(len(filelines))
    
    for i in range(0, len(filelines)):
        print(i)
        fileline = filelines[i]
        document = '\n'.join(fileline)
        result = cls.evaluate_single_document(document)
        print(result)  
    cls.network.close()       
