'''
Created on Oct 3, 2018

@author: g.werner
'''

from aylienapiclient import textapi

def convert_scale(sentiment, confidence):
    if sentiment == 'positive':
        return confidence
    elif sentiment == 'negative':
        return -confidence
    elif sentiment == 'neutral':
        return 0
    return float('NaN')
    
class AylienSentiment(object):

    def __init__(self):
        self.c = textapi.Client("ef4bd34e", "96f1defa100d38c77407df4d03a55d5b")
        self.server_on = False

    def config(self):
        try:
            self.server_on = True
        except:
            pass
        
    def evaluate_single_document(self, text, mode):
        if mode == 'document':
            s = self.c.Sentiment({'text': text})
            return [convert_scale(s['polarity'], s['polarity_confidence'])]
        if mode == 'entity':
            elsa = self.c.Elsa({'text': text})
            
            to_return = []
            entity_dict = {}
            
            for entity in elsa['entities']:
                overall = entity['overall_sentiment']
                entity_dict[entity['mentions'][0]['text']] = (convert_scale(overall['polarity'], overall['confidence']))
            to_return.append(entity_dict)
            return to_return
        return []
        
if __name__ == '__main__':
    text ='Amazon is great, but Tesla is really horrible.  There is truth to that.'
    ays = AylienSentiment()
    ays.config('document')
    result = ays.evaluate_single_document(text)
    print(result)