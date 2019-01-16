'''
Created on Oct 2, 2018

@author: g.werner
'''

class Config(object):
    APP_NAME = 'SentimentService'
    
class DevelopmentConfig(Config):
    STANFORD_LOCATION="stanford-corenlp-full-2018-02-27"
    STANFORD_SERVER="http://localhost"
    STANFORD_PORT=9000
