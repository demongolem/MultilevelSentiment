'''
Created on Dec 3, 2018

@author: g.werner
'''

import CompositeSentiment

def output_to_disc(score, mode, alg):
    pth = 'alternatives'
    with open(pth + '/' + 'score_' + alg + '_' + str(mode) + '.txt', 'w') as f:
        for key in score:
            f.write(key + '\t' + str(score[key]) + '\n')

def main():
    composite = CompositeSentiment.CompositeSentiment()
    start_doc = 0
    duration_doc = -1    
    algorithms = ['gd','lgbs', 'sm']
    
    for annotator_mode in range(1, 512):
        for algorithm in algorithms:
            # train call
            new_weights = composite.train_parameters(composite.convert_annotator_code(annotator_mode), algorithm, start_doc, duration_doc)
            CompositeSentiment.write_weights_to_file(new_weights, 'optimal_weights.txt')
            # eval call
            trained_weights = CompositeSentiment.read_weights('optimal_weights.txt', True)
            score = composite.score(trained_weights)
            # output the score to disc before proceeding
            output_to_disc(score, annotator_mode, algorithm)
            # refresh the weights back to their initial "guess"
            composite.init_weights()

if __name__ == '__main__':
    main()