'''
Created on Sep 23, 2018

@author: g.werner
'''
import Config
from flask import Flask
from flask import request
import json
import pathlib
import spacy
import sys
from AylienSentiment import AylienSentiment
from CharLSTMSentiment import CharLSTMSentiment
from CompositeSentiment import CompositeSentiment
import FinanceSentiment
from GoogleCloudSentiment import GoogleCloudSentiment
from SpacySentiment import SentimentAnalyser
import SpacySentiment
from StanfordSentiment import StanfordSentiment
import TweepySentiment
import VaderSentiment

application = Flask(__name__)

# This line would accomplish lazy loading.  But, according to GIT issue #2, we don't want it.
#application.before_first_request(init)

stanford_sentiment = StanfordSentiment()
google_sentiment = GoogleCloudSentiment()
aylien_sentiment = AylienSentiment()
char_lstm_sentiment = CharLSTMSentiment()
composite_sentiment = CompositeSentiment()

### Handle command line arguments ###
# dev (default), staging, production
env = sys.argv[1] if len(sys.argv) > 1 else 'dev'

if env == 'dev':
    sent_config = Config.DevelopmentConfig()
else:
    raise ValueError('Invalid environment name ' + env)

stanford_sentiment.config(sent_config)
google_sentiment.config()
aylien_sentiment.config()
char_lstm_sentiment.config(sent_config)

def init():
    print('Loading Spacy Vectors')
    global nlp, sa
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe(SentimentAnalyser.load(pathlib.Path('model'), nlp, max_length=100))

init()

@application.route("/spacy", methods = ['GET', 'POST'])
def get_spacy_sentiment():
    if request.method == 'GET':
        text = [request.args.get('texts')]
    elif request.method == 'POST':
        text = [request.form['texts']]
    else:
        return ('Unknown method!!!')
    return str(compute_spacy_sentiment(text))

def compute_spacy_sentiment(text):
    global nlp
    return SpacySentiment.evaluate_without_labels(nlp, text)

@application.route("/vader", methods = ['GET', 'POST'])
def get_vader_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
    elif request.method == 'POST':
        text = request.form['texts']
    else:
        return ('Unknown method!!!')
    return str(compute_vader_sentiment(text))

def compute_vader_sentiment(text):
    return VaderSentiment.evaluate_single_document(text)

@application.route("/tweepy", methods = ['GET', 'POST'])
def get_tweepy_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
    elif request.method == 'POST':
        text = request.form['texts']
    else:
        return ('Unknown method!!!')
    return str(compute_tweepy_sentiment(text))

def compute_tweepy_sentiment(text):
    return TweepySentiment.evaluate_single_document(text)

@application.route("/finance", methods = ['GET', 'POST'])
def get_finance_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
    elif request.method == 'POST':
        text = request.form['texts']
    else:
        return 'Unknown method!'
    (positive, negative) = compute_finance_sentiment(text)
    return str(positive) + '\t' + str(negative)    

def compute_finance_sentiment(text):
    return FinanceSentiment.evaluate_single_document(text)

@application.route("/stanford", methods = ['GET', 'POST'])
def get_stanford_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
        mode = request.args.get('mode')
    elif request.method == 'POST':
        text = request.form['texts']
        mode = request.form['mode']
    else:
        return ('Unknown method!!!')
    
    polarity = compute_stanford_sentiment(text, mode)
    return json.dumps(polarity) if polarity is not None else "Stanford server is currently down.  Please try again later!"

def compute_stanford_sentiment(text, mode):
    # this will fail if there is no running server as dictated by the Config.py setting
    return stanford_sentiment.evaluate_single_document(text, mode)

@application.route("/google", methods = ['GET', 'POST'])
def get_google_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
        mode = request.args.get('mode')
    elif request.method == 'POST':
        text = request.form['texts']
        mode = request.form['mode']
    else:
        return ('Unknown method!!!')
    return json.dumps(compute_google_sentiment(text, mode))

def compute_google_sentiment(text, mode):
    return google_sentiment.evaluate_single_document(text, mode)

@application.route("/aylien", methods = ['GET', 'POST'])
def get_aylien_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
        mode = request.args.get('mode')
    elif request.method == 'POST':
        text = request.form['texts']
        mode = request.form['mode']
    else:
        return ('Unknown method!!!')
    return json.dumps(compute_aylien_sentiment(text, mode))

def compute_aylien_sentiment(text, mode):
    return aylien_sentiment.evaluate_single_document(text, mode)

@application.route("/charlstm", methods = ['GET', 'POST'])
def get_char_lstm_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
        mode = request.args.get('mode')
    elif request.method == 'POST':
        text = request.form['texts']
        mode = request.form['mode']
    else:
        return ('Unknown method!!!')
    return json.dumps(compute_lstm_sentiment(text, mode))

def compute_lstm_sentiment(text, mode):
    return char_lstm_sentiment.evaluate_single_document(text, mode)

@application.route("/composite", methods = ['GET', 'POST'])
# get combined score
def get_composite_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
        preset = request.args.get('preset')
    elif request.method == 'POST':
        text = request.form['texts']
        preset = request.form['preset']
    (seven_scores, financial_scores, annotator_mode) = compute_composite_sentiment(text, preset)
    return json.dumps(composite_blend(seven_scores, financial_scores, annotator_mode))

@application.route("/composite2", methods = ['GET', 'POST'])
# get individual value
def get_composite2_sentiment():
    if request.method == 'GET':
        text = request.args.get('texts')
        preset = request.args.get('preset')
    elif request.method == 'POST':
        text = request.form['texts']
        preset = request.form['preset']
    (seven_scores, financial_scores, annotator_mode) = compute_composite_sentiment(text, preset)
    to_return = seven_scores
    to_return = {**to_return, **financial_scores}
    return json.dumps(to_return)

def compute_composite_sentiment(text, preset = 'all'):
    # preset is either a list of annotators or a shortcut repreresentation of a list of annotators
    if preset == 'rule_based':
        seven_scores = {}
        # add the relevant annotators which given an answer in range [-1, 1]
        seven_scores['vader'] = compute_vader_sentiment(text)
        seven_scores['tweepy'] = compute_tweepy_sentiment(text)
        # finally add the financial advice
        financial_scores = {}
        fs = compute_finance_sentiment(text)
        financial_scores['finance_pos'] = fs[0]
        financial_scores['finance_neg'] = fs[1]
        return(seven_scores, financial_scores, 390)
    elif preset == 'no_lstm':
        seven_scores = {}
        # add the seven annotators which given an answer in range [-1, 1]
        seven_scores['spacy'] = compute_spacy_sentiment([text])
        seven_scores['vader'] = compute_vader_sentiment(text)
        seven_scores['tweepy'] = compute_tweepy_sentiment(text)
        seven_scores['stanford'] = compute_stanford_sentiment(text)[0]
        seven_scores['google'] = compute_google_sentiment(text)[0]
        seven_scores['aylien'] = compute_aylien_sentiment(text)[0]
        # finally add the financial advice
        financial_scores = {}
        fs = compute_finance_sentiment(text)
        financial_scores['finance_pos'] = fs[0]
        financial_scores['finance_neg'] = fs[1]
        return(seven_scores, financial_scores, 447)
    elif preset == 'tw_va_go':
        seven_scores = {}
        # add the seven annotators which given an answer in range [-1, 1]
        seven_scores['vader'] = compute_vader_sentiment(text)
        seven_scores['tweepy'] = compute_tweepy_sentiment(text)
        seven_scores['google'] = compute_google_sentiment(text)[0]
        # finally add the financial advice
        financial_scores = {}
        fs = compute_finance_sentiment(text)
        financial_scores['finance_pos'] = fs[0]
        financial_scores['finance_neg'] = fs[1]
        return(seven_scores, financial_scores, 406)
    elif preset == 'all':
        seven_scores = {}
        # add the seven annotators which given an answer in range [-1, 1]
        seven_scores['spacy'] = compute_spacy_sentiment([text])
        seven_scores['vader'] = compute_vader_sentiment(text)
        seven_scores['tweepy'] = compute_tweepy_sentiment(text)
        seven_scores['stanford'] = compute_stanford_sentiment(text)[0]
        seven_scores['google'] = compute_google_sentiment(text)[0]
        seven_scores['aylien'] = compute_aylien_sentiment(text)[0]
        seven_scores['charlstm'] = compute_lstm_sentiment(text)[0]
        # finally add the financial advice
        financial_scores = {}
        fs = compute_finance_sentiment(text)
        financial_scores['finance_pos'] = fs[0]
        financial_scores['finance_neg'] = fs[1]
        return(seven_scores, financial_scores, 511)
    else:
        # the fallback assumes a list of annotators
        chosen_annotators = set(preset.split(','))
        seven_scores = {}
        annotator_mode = 0
        # add the seven annotators which given an answer in range [-1, 1]
        if 'spacy' in chosen_annotators:
            seven_scores['spacy'] = compute_spacy_sentiment([text])
            annotator_mode += 1
        if 'vader' in chosen_annotators:
            seven_scores['vader'] = compute_vader_sentiment(text)
            annotator_mode += 2
        if 'tweepy' in chosen_annotators: 
            seven_scores['tweepy'] = compute_tweepy_sentiment(text)
            annotator_mode += 4
        if 'stanford' in chosen_annotators:
            seven_scores['stanford'] = compute_stanford_sentiment(text)[0]
            annotator_mode += 8
        if 'google' in chosen_annotators:
            seven_scores['google'] = compute_google_sentiment(text)[0]
            annotator_mode += 16
        if 'aylien' in chosen_annotators:
            seven_scores['aylien'] = compute_aylien_sentiment(text)[0]
            annotator_mode += 32
        if 'charlstm' in chosen_annotators:
            seven_scores['charlstm'] = compute_lstm_sentiment(text)[0]
            annotator_mode += 64
        # finally add the financial advice
        financial_scores = {}
        if 'finance_pos' in chosen_annotators or 'finance_neg' in chosen_annotators:
            returned_value = compute_finance_sentiment(text)
            if 'finance_pos' in chosen_annotators:
                financial_scores['finance_pos'] = returned_value[0]
                annotator_mode += 128
            if 'finance_neg' in chosen_annotators:
                financial_scores['finance_neg'] = returned_value[1]
                annotator_mode += 256
        return(seven_scores, financial_scores, annotator_mode)

def composite_blend(seven_scores, financial_scores, annotator_mode):
    polarity = composite_sentiment.evaluate_single_document(seven_scores, financial_scores, annotator_mode)
    return polarity

@application.route("/list", methods = ['GET'])
def get_endpoints():    
    return json.dumps([{'spacy':'Document based sentiment', 'vader':'Document based sentiment', 'tweepy':'Document based sentiment',
             'finance':'Document based sentiment', 'stanford':'Document and Sentence based sentiment',
             'google':'Document, sentence and entity based sentiment', 'aylien':'Document and Entity based sentiment', 
             'charlstm':'Document, sentence based sentiment', 'composite':'Document based sentiment', 
             'list':'List all endpoints', '':'Health Check'}])

@application.route("/")
def healthcheck():
    return "Still alive"

if __name__ == '__main__':
    application.run(port='8086', threaded=False)
