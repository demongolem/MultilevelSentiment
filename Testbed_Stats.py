'''
Created on Nov 1, 2018

@author: g.werner
'''
import collections
import csv
import math
import statistics
import sys

## data structure if dict(string->dict(string->pair))

def create_object(csv_file):
    file_dict = collections.OrderedDict()
    with open(csv_file) as csv_file_obj:
        csv_reader = csv.reader(csv_file_obj, delimiter=',')
        header = True
        for row in csv_reader:
            if header:
                # this is the header
                print(f'Column names are {", ".join(row)}')
                header = False
            else:
                filename = row[0]
                method = row[1]
                time = float(row[2])
                polarity = float(row[3])
                #positive_terms = float(row(4))
                #negative_terms = float(row(5))
                if filename not in file_dict:
                    file_dict[filename] = collections.OrderedDict()
    
                if math.isnan(polarity):
                    continue
                file_dict[filename][method] = (time, polarity)
    
    return file_dict

def mean(method_dict, dimension):
    summ = 0.0
    count = 0.0
    if 'time' == dimension:
        for key in method_dict:
            count += 1.0
            summ += method_dict[key][0]
        return summ / count
    elif 'score' == dimension:
        for key in method_dict:
            count += 1.0
            summ += method_dict[key][1]
        return summ / count
    elif 'none' == dimension:
        for value in method_dict:
            count += 1.0
            summ += value
        return summ / count            
    else:
        return(float('nan'))

def min(method_dict, dimension):
    min_value = sys.float_info.max
    if 'none' == dimension:
        for value in method_dict:
            if value < min_value:
                min_value = value
        return min_value
    else:
        return(float('nan'))

def max(method_dict, dimension):
    max_value = sys.float_info.min
    if 'none' == dimension:
        for value in method_dict:
            if value > max_value:
                max_value = value
        return max_value
    else:
        return(float('nan'))

def median(method_dict, dimension):
    if 'none' == dimension:
        return statistics.median(method_dict)
    else:
        return(float('nan'))    
    

def var(method_dict, dimension):
    meann = mean(method_dict, dimension)
    sum_sq = 0.0
    if 'time' == dimension:
        for key in method_dict:
            sum_sq += (method_dict[key][0] - meann) ** 2.0
        return sum_sq / (len(method_dict) - 1)
    elif 'score' == dimension:
        for key in method_dict:
            sum_sq += (method_dict[key][1] - meann) ** 2.0
        return sum_sq / (len(method_dict) - 1)
    elif 'none' == dimension:
        for value in method_dict:
            sum_sq += (value - meann) ** 2.0
        return sum_sq / (len(method_dict) - 1)
    else:
        return(float('nan'))
    
def stdev(method_dict, dimension):
    varr = var(method_dict, dimension)
    return math.sqrt(varr)

def skew(method_dict, dimension):
    stdevv = stdev(method_dict, dimension)
    meann = mean(method_dict, dimension)
    sum_cu = 0.0
    if 'time' == dimension:
        for key in method_dict:
            sum_cu += (method_dict[key][0] - meann) ** 3.0
    elif 'score' == dimension:
        for key in method_dict:
            sum_cu += (method_dict[key][1] - meann) ** 3.0
    elif 'none' == dimension:
        for value in method_dict:
            sum_cu += (value - meann) ** 3.0
    else:
        return(float('nan'))
    return float('nan') if stdevv == 0 else sum_cu / stdevv ** 3

def kurtosis(method_dict, dimension):
    stdevv = stdev(method_dict, dimension)
    meann = mean(method_dict, dimension)
    sum_biqu = 0.0
    if 'time' == dimension:
        for key in method_dict:
            sum_biqu += (method_dict[key][0] - meann) ** 4.0
    elif 'score' == dimension:
        for key in method_dict:
            sum_biqu += (method_dict[key][1] - meann) ** 4.0
    elif 'none' == dimension:
        for value in method_dict:
            sum_biqu += (value - meann) ** 4.0
    else:
        return(float('nan'))
    return float('nan') if stdevv == 0 else sum_biqu / stdevv ** 4

def stderr(method_dict, dimension):
    stdevv = stdev(method_dict, dimension)
    count = len(method_dict)
    return stdevv / (count ** 0.5)

def bias(method_dict, dimension):
    meann = mean(method_dict, dimension)
    summ= 0.0
    if 'time' == dimension:
        for key in method_dict:
            summ += (method_dict[key][0] - meann)
        return summ
    elif 'score' == dimension:
        for key in method_dict:
            summ += (method_dict[key][1] - meann)
        return summ
    elif 'none' == dimension:
        for value in method_dict:
            summ += (value - meann)
        return summ    
    else:
        return(float('nan'))

def mse(method_dict, dimension):
    stderrr = stderr(method_dict, dimension)
    biass = bias(method_dict, dimension)
    return stderrr ** 2 + biass ** 2

def fill_in_stats(file_dict):
    stats_dict = collections.OrderedDict()
    for key in file_dict:
        stats_dict[key] = collections.OrderedDict()
        stats_dict[key]['mean_time'] = mean(file_dict[key],'time')
        stats_dict[key]['var_time'] = var(file_dict[key],'time')
        stats_dict[key]['stdev_time'] = stdev(file_dict[key],'time')
        stats_dict[key]['skew_time'] = skew(file_dict[key],'time')
        stats_dict[key]['kurtosis_time'] = kurtosis(file_dict[key],'time')
        stats_dict[key]['bias_time'] = bias(file_dict[key],'time')
        stats_dict[key]['stderr_time'] = stderr(file_dict[key],'time')
        stats_dict[key]['mse_time'] = mse(file_dict[key],'time')
        stats_dict[key]['mean_score'] = mean(file_dict[key],'score')
        stats_dict[key]['var_score'] = var(file_dict[key],'score')        
        stats_dict[key]['stdev_score'] = stdev(file_dict[key],'score')
        stats_dict[key]['skew_score'] = skew(file_dict[key],'score')
        stats_dict[key]['kurtosis_score'] = kurtosis(file_dict[key],'score')
        stats_dict[key]['bias_score'] = bias(file_dict[key],'score')  
        stats_dict[key]['stderr_score'] = stderr(file_dict[key],'score')
        stats_dict[key]['mse_score'] = mse(file_dict[key],'score')          
    return stats_dict
 
def fill_in_annotator_stats(file_dict):
    sample_dict = collections.OrderedDict()
    stats_dict = collections.OrderedDict()
    for key in file_dict:
        for key2 in file_dict[key]:
            if key2 not in sample_dict:
                sample_dict[key2] = []
            data_pair = file_dict[key][key2]
            time = data_pair[0]
            sent = data_pair[1]
            sample_dict[key2].append(sent)
    for key in sample_dict:
        stats_dict[key] = collections.OrderedDict()
        stats_dict[key]['mean_score'] = mean(sample_dict[key], 'none')
        stats_dict[key]['median'] = median(sample_dict[key], 'none')
        stats_dict[key]['min'] = min(sample_dict[key], 'none')
        stats_dict[key]['max'] = max(sample_dict[key], 'none')
        stats_dict[key]['var_score'] = var(sample_dict[key], 'none')        
        stats_dict[key]['stdev_score'] = stdev(sample_dict[key], 'none')
        stats_dict[key]['skew_score'] = skew(sample_dict[key], 'none')
        stats_dict[key]['kurtosis_score'] = kurtosis(sample_dict[key], 'none')
        stats_dict[key]['bias_score'] = bias(sample_dict[key], 'none')  
        stats_dict[key]['stderr_score'] = stderr(sample_dict[key], 'none')
        stats_dict[key]['mse_score'] = mse(sample_dict[key], 'none')
    print(stats_dict)
    return stats_dict
 
def write_to_csv(filename, stats_dict):
    with open(filename, 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(["File", "mean_time", "var_time", "stdev_time", "skew_time", "kurtosis_time", "bias_time", "stderr_time", "mse_time", "mean_score", "var_score", "stdev_score", "skew_score", "kurtosis_score", "bias_score", "stderr_score", "mse_score"])
        for key in stats_dict:
            the_list = [key]
            for stats_key in stats_dict[key]:
                the_list.append(stats_dict[key][stats_key])
            writer.writerow(the_list)
         
def main(csv_file):
    file_dict = create_object(csv_file)
    stats_dict = fill_in_annotator_stats(file_dict)
    #stats_dict = fill_in_stats(file_dict)
    #write_to_csv('master_test_stats.csv', stats_dict)

if __name__ == '__main__':
    csv_file = 'master_test_results.csv'
    main(csv_file)