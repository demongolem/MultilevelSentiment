'''
Created on Nov 13, 2018

@author: g.werner
'''

from collections import OrderedDict
import json
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import array
from pathlib import Path
import scipy.optimize as opt
import statsmodels.api as sm
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

def_gd_iters = 10000
def_gd_alpha = 0.05

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))    

# weight training method 1
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)

    return theta

# weight training method 2
def l_bfgs_b(X, y, theta, mybounds):
    info = opt.fmin_l_bfgs_b(func, x0=theta, args=(X,y), approx_grad=True, factr = 10.0, maxls = 100, bounds = mybounds)
    return info[0]

def func(params, *args):
    X = args[0]
    y = args[1]
    theta = params
    step1 = X @ theta.T
    diff = step1.reshape(len(step1),1) - y
    error = diff
    return sum(error**2)

# weight training method 3
def reg_m(y, x):
    X = x
    results = sm.OLS(y, X).fit()
    return results.params

def sentiment_to_numeric(sentiment_list):
    to_return = []
    for item in sentiment_list:
        if item == 'Very Positive':
            to_return.append([1.0])
        elif item == 'Positive':
            to_return.append([0.5])
        elif item == 'Negative':
            to_return.append([-0.5])
        elif item == 'Very Negative':
            to_return.append([-1.0])
        else:
            to_return.append([0.0])
    return to_return

def generate_composite_list(full_dict, annotators):
    new_dict = OrderedDict()
    for annotator in annotators:
        if annotator in full_dict:
            new_dict[annotator] = full_dict[annotator]
    return new_dict

def write_weights_to_file(weights, outfile):
    dataile_id = open(outfile, 'w+')
    np.savetxt(dataile_id, weights, delimiter=" ", fmt="%s")
    dataile_id.close()

def read_weights(filename, is_vertical):
    weights = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        if is_vertical:
            for line in lines:
                parts = line.rstrip().split(' ')
                weights[parts[0]] = float(parts[1])
        else:
            line = lines[0].rstrip()
            parts = line.split(' ')
            for part in parts:
                weights.append(float(part))
    return weights

def put_method_to_disk(filename, document, values):
    with open(filename, 'a') as f:
        f.write(document + '\t' + '\t'.join(str(v) + " " + str(values[v]) for v in values) + '\n')
 
class CompositeSentiment(object):

    def __init__(self, debugging = False):
        self.debugging = debugging
        self.annotator_list = ['spacy','vader','tweepy','stanford','google','aylien','charlstm','finance_pos','finance_neg']
        # these weights will be overridden at a later time
        self.init_weights()
                    
        self.composite_scores = {}
        my_file = Path("composite2_individual_scores_all.txt")
        if my_file.is_file():
            with my_file.open() as f: 
                lines = f.readlines()
                for line in lines:
                    parts = line.split('\t')
                    self.composite_scores[parts[0]] = {}
                    for j in range(1, len(parts)):
                        inner_parts = parts[j].split(' ')
                        self.composite_scores[parts[0]][inner_parts[0]] = float(inner_parts[1])

    def init_weights(self):
        self.weights = {}
        self.mybounds = {}
        for i in range(0, 7):
            self.weights[self.annotator_list[i]] = 1.0 / 7.0
            self.mybounds[self.annotator_list[i]] = (0.0, 1.0)
        for i in range(7, 8):
            self.weights[self.annotator_list[i]] = 0.05
            self.mybounds[self.annotator_list[i]] = (None, None)
        for i in range(8, 9):
            self.weights[self.annotator_list[i]] = -0.05
            self.mybounds[self.annotator_list[i]] = (None, None)

    # returns dict file -> overall sentiment score
    def score(self, weights_dict):
        score_dict = {}
        minz = float('Inf')
        maxz = -float('Inf')
        #something like this, but we need the list of annotators, given us by the weights
        annotators = list(weights_dict.keys())
        for key in self.composite_scores:
            # value_dict is a dict annotator => value
            value_dict = self.composite_scores[key]
            reduced_composite = generate_composite_list(value_dict, annotators)
            running_score = 0.0
            for ikey in annotators:
                running_score += weights_dict[ikey] * reduced_composite[ikey]
            score_dict[key] = running_score
            if running_score < minz:
                minz = running_score
            if running_score > maxz:
                maxz = running_score
        print('Min is ' + str(minz) + ' and Max is ' + str(maxz))
        return score_dict

    def convert_annotator_code(self, code):
        to_return = []
        if code & 1:
            to_return.append(self.annotator_list[0])
        if code & 2:
            to_return.append(self.annotator_list[1])
        if code & 4:
            to_return.append(self.annotator_list[2])
        if code & 8:
            to_return.append(self.annotator_list[3])
        if code & 16:
            to_return.append(self.annotator_list[4])
        if code & 32:
            to_return.append(self.annotator_list[5])
        if code & 64:
            to_return.append(self.annotator_list[6])
        if code & 128:
            to_return.append(self.annotator_list[7])
        if code & 256:
            to_return.append(self.annotator_list[8])
        return to_return
    
    def lookupMethodText(self, method, document, text, preset):
        while True:
            endpoint = 'http://localhost:8086/'
            
            full_endpoint = endpoint + method
            
            post_fields = {'texts': text, 'preset': ','.join(preset)}     # Set POST fields here
            request = Request(full_endpoint, urlencode(post_fields).encode())
            try:
                obj = urlopen(request).read()
            except HTTPError as timeout:
                print('Retrying: ' + str(timeout))
                continue
            json_representation = json.loads(obj)
            if 'True' == self.debugging:
                put_method_to_disk(method + '_individual_scores.txt', document, json_representation)
            return json_representation

    def evaluate_single_document(self, seven_scores, financial_scores, annotator_mode):
        #setup correct weights first
        with open('weights/optimal_weights_' + str(annotator_mode) + '.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.rstrip().split(' ')
                self.weights[parts[0]] = float(parts[1])

        #now that we are sure about the weights, we can proceed to calculate the final sentiment value
        to_return = 0.0
        for key in seven_scores:
            to_return += self.weights[key] * seven_scores[key]
        
        to_return /= len(seven_scores)
        
        for key in financial_scores:
            to_return += financial_scores[key] * self.weights[key]
        
        return to_return
        
    def train_parameters(self, annotators, algorithm, start, duration):
        # section 0, create numpy array of chosen annotators
        labels = np.array(annotators)
        reduced_weights = generate_composite_list(self.weights, annotators) 
        reduced_bounds = generate_composite_list(self.mybounds, annotators)

        # section 1, read gold information
        gold_sentiments = OrderedDict()
        # gold directory is hardcoded right now, if you notice.  it could change later
        train_sent_dir = 'input/train_sent';
        onlyfiles = [f for f in listdir(train_sent_dir) if isfile(join(train_sent_dir, f))]
        if duration == -1:
            duration = len(onlyfiles) - start
        count = 0
        for onlyfile in onlyfiles:
            if count < start:
                count += 1
                continue
            elif count >= start + duration:
                break
            with open(join(train_sent_dir, onlyfile)) as f:
                lines = f.readlines()
                gold_sentiments[join(train_sent_dir, onlyfile)] = lines[0].rstrip()
                count += 1
        print('Done reading gold files')
        
        #section 2, get the values for the same documents for the chosen annotators
        count = 0
        predicted_sentiments = OrderedDict()
        # once again, hardcoded.  it could be changed later on
        train_raw_dir = 'input/train_raw';
        onlyrawfiles = [f for f in listdir(train_raw_dir) if isfile(join(train_raw_dir, f))]
        for onlyfile in onlyrawfiles:
            if count < start:
                count += 1
                continue
            elif count >= start + duration:
                count = 0
                break            
            with open(join(train_raw_dir, onlyfile)) as f:
                print('Predicting '+ onlyfile)
                text = f.read()
                if onlyfile in self.composite_scores:
                    print('Known '  + onlyfile)
                    reduced_composite = generate_composite_list(self.composite_scores[onlyfile], annotators)
                    predicted_sentiments[onlyfile] = list(reduced_composite.values())
                else:
                    print('Unknown ' + onlyfile)
                    relevant =  self.lookupMethodText('composite2', onlyfile, text, annotators)
                    predicted_sentiments[onlyfile] = list(relevant.values())
                count += 1
        
        #section 3, optimize for the best constants        
        X = array(list(predicted_sentiments.values()))
        y = array(sentiment_to_numeric(list(gold_sentiments.values())))
        theta = array(list(reduced_weights.values()))
        if algorithm == 'gd':
            # need to redefine for gd, whoops!
            theta = array([list(reduced_weights.values())])
            best_answer = gradientDescent(X, y, theta, def_gd_iters, def_gd_alpha)
            best_answer = np.vstack((labels, best_answer)).T
        elif algorithm == 'lgbs':
            best_answer =  l_bfgs_b(X, y, theta, list(reduced_bounds.values()))
            best_answer = np.vstack((labels, best_answer)).T
        elif algorithm == 'sm':
            best_answer =  reg_m(y, X)
            best_answer = np.vstack((labels, best_answer)).T
        else:
            best_answer = None         
        return best_answer
    
if __name__ == '__main__':
    if len(sys.argv) != 7:
        print('Arguments: train/test annotator_mode algorithm begin duration debug?')
        sys.exit()
    mode = sys.argv[1]
    # 511 would be "all"
    annotator_mode = int(sys.argv[2])
    algorithm = sys.argv[3]
    start_doc = int(sys.argv[4])
    duration_doc = int(sys.argv[5])
    debugging = sys.argv[6]
    composite = CompositeSentiment(debugging)
    texts = 'I think that that movie was really good.  Wonderful acting plus great special effects made for an entertaining evening.'
    seven_scores = {}
    financial_scores = {}
    for i in range(0, 7):
        seven_scores[composite.annotator_list[i]] = 0.0
    for i in range(7, 9):
        financial_scores[composite.annotator_list[i]] = 0.0
    if mode == 'train':
        print('Training new model')
        new_weights = composite.train_parameters(composite.convert_annotator_code(annotator_mode), algorithm, start_doc, duration_doc)
        write_weights_to_file(new_weights, 'weights/optimal_weights_' + str(annotator_mode) + '.txt')
    elif mode == 'eval':
        print('Using weights to score training set')
        # i don't know that this filename should be hardcoded
        trained_weights = read_weights('weights/optimal_weights_' + str(annotator_mode) + '.txt', True)
        score = composite.score(trained_weights)
        print(score)
    else:
        print('Classifying single document')
        final_score = composite.evaluate_single_document(seven_scores, financial_scores)
        print(final_score)