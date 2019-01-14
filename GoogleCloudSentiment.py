'''
Created on Oct 2, 2018

@author: g.werner
'''
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import six
import sys

def convert_scale(sentiment):
    score = sentiment.score
    # magnitude disambiguates neutral vs cancel out when score are around 0
    # we have no convention currently to make use of this value, so the score stands as our output
    magnitude = sentiment.magnitude
    return score

class GoogleCloudSentiment(object):

    def __init__(self):
        self.server_on = False
        self.client = language.LanguageServiceClient()

    def config(self):
        try:
            self.server_on = True
        except:
            pass

    def evaluate_single_document(self, text, mode):
        
        """Detects entity sentiment in the provided text."""
        
        if isinstance(text, six.binary_type):
            text = text.decode('utf-8')
    
        document = types.Document(
            content=text.encode('utf-8'),
            type=enums.Document.Type.PLAIN_TEXT,
            language='en'
        )
    
    
        # Detect and send native Python encoding to receive correct word offsets.
        encoding = enums.EncodingType.UTF32
        if sys.maxunicode == 65535:
            encoding = enums.EncodingType.UTF16
    
        if mode == 'document' or mode == 'sentence':
            document_result = self.client.analyze_sentiment(document, encoding)
            if mode == 'document':
                sentiment = document_result.document_sentiment
                return [convert_scale(sentiment)]
            else:
                to_return = []
                for sentence in document_result.sentences:
                    to_return.append(convert_scale(sentence.sentiment)) 
                return to_return
                       
        if mode == 'entity':
            entity_result = self.client.analyze_entity_sentiment(document, encoding)
    
            to_return = []
            entity_dict = {}
    
            for entity in entity_result.entities:
                entity_dict[entity.name] = convert_scale(entity.sentiment)
    
            to_return.append(entity_dict)
             
            return to_return
        
        return []
        
if __name__ == '__main__':
    text ='Amazon is great, but Tesla is really horrible.  There is truth to that.'
    gcs = GoogleCloudSentiment()
    gcs.config('entity')
    result = gcs.evaluate_single_document(text)
    print(result)