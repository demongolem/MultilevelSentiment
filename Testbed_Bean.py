'''
Created on Oct 29, 2018

@author: g.werner
'''
def decode_line_pair(lines):
    to_return = {}
    if len(lines) != 2:
        return None
    to_return['test_file'] = lines[0].strip() 
    tabs = lines[1].count('\t')
    tab_split = lines[1].split('\t')
    if tabs == 3:
        to_return['method'] = tab_split[1]
        to_return['polarity'] = float(tab_split[2].replace('[','').replace(']',''))
        to_return['positives'] = float('NaN')
        to_return['negatives'] = float('NaN')
        to_return['time'] = float(tab_split[3])
    elif tabs == 4:
        to_return['method'] = tab_split[1]
        to_return['polarity'] = float('NaN')
        to_return['positives'] = int(tab_split[2])
        to_return['negatives'] = int(tab_split[3])
        to_return['time'] = float(tab_split[4])
    else:
        return None
    return to_return