'''
Created on Oct 24, 2018

@author: g.werner
'''
# Filename -> Method -> (Score,Time)

from os import listdir
from os.path import isfile, join
import sys
import time
from time import sleep
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

side_effect = []

AYLIEN_START = 2800
AYLIEN_INCREMENT= 100

SPACY_START = 0
SPACY_INCREMENT = 5143

def fetch_files(directory):
    global side_effect
    filelines = []
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    for onlyfile in onlyfiles:
        side_effect.append(onlyfile)
        with open(join(directory, onlyfile), 'r', encoding="utf-8") as f:
            filelines.append(f.readlines())
    return filelines

def perform(method, fileline):
    while True:
        document = '\n'.join(fileline)
        endpoint = 'http://localhost:8086/'
        
        full_endpoint = endpoint + method
        
        post_fields = {'texts': document}     # Set POST fields here
        request = Request(full_endpoint, urlencode(post_fields).encode())
        start_time = time.time()
        try:
            json = urlopen(request).read().decode()
        except HTTPError as timeout:
            print('Retrying: ' + str(timeout))
            continue
        end_time = time.time()
        return (json, end_time - start_time)

def output_results(output_file, timing_dict):
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_dict in timing_dict:
            f.write(file_dict + '\n')
            for entry in timing_dict[file_dict]:
                f.write('\t' + entry + '\t' + str(timing_dict[file_dict][entry][0]) + '\t' + str(timing_dict[file_dict][entry][1]) + '\n')

# start and length are currently only used for charlstm
def main(directory, output_file, start, length, methods):
    global side_effect
    
    print('Directory is ' + directory)
    print('Output File is ' + output_file)
    print('Start is ' + str(start))
    print('Length is ' + str(length))
    print('Methods are ' + ','.join(methods))

    timing_dict = {}

    print('Fetching files')
    start_time = time.time()
    filelines = fetch_files(directory)
    end_time = time.time()
    print(str(len(filelines)) + ' files fetched in ' + str(end_time - start_time) + ' seconds')

    for method in methods:
        # Risk a 429 error if we have too many requests per second
        print('Finding sentiment using ' + method)
        if method == 'aylien':
            start_index = AYLIEN_START
            end_index = start_index + AYLIEN_INCREMENT
        elif method == 'spacy':
            start_index = SPACY_START
            end_index = start_index + SPACY_INCREMENT
        elif method == 'charlstm':
            start_index = start
            end_index = start_index + length
        else:
            start_index = 0
            end_index = len(filelines)
        for i in range(start_index, end_index):
            if 'google' == method:
                sleep(0.100)
            elif 'aylien' == method:
                sleep(1.000)          
            fileline = filelines[i]
            filename = side_effect[i]
            print(str(i) + ' ' + filename)
            result = perform(method, fileline)
            if filename not in timing_dict:
                timing_dict[filename] = {}
            timing_dict[filename][method] = result
        print('Done sentiment using ' + method)

    output_results(output_file, timing_dict)

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: Testbed <source_dir> <output_file> <start> <length> <methods>')
    else:
        main(sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5:])