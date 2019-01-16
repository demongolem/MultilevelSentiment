# MultilevelSentiment
Perform sentiment analysis at the document, sentence, or entity level.

Make sure that you unzip the models in datasets before you try to run the application.

The program will attempt to start a local Stanford server if an existing Stanford server is not running as configured in Config.py.

This program depends upon Stanford CoreNLP.  As such it is highly recommended to use 3.9.1 right now.  This is a firm requirement when running Agent Annotator on top of it.  

The stanfordcorenlp library used was built from the master of https://github.com/Lynten/stanford-corenlp (using setup.py) and not from the commonly available 3.9.1.1 distribution.  We need the custom version because it gives us the max_retries which was put into place after the distribution release.
