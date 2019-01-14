'''
Created on Dec 3, 2018

@author: g.werner
'''

from os import listdir
from os.path import isfile, join
import xlsxwriter

def write_to_xls(d):
    workbook = xlsxwriter.Workbook('combo_analysis.xlsx')
    worksheet = workbook.add_worksheet()

    row = -2
    max_col = 0

    #Remember, we are trying to transpose the matrix!
    for key in d.keys():
        row += 2
        col = 0
        worksheet.write(col, row, key)
        for ikey in d[key]:
            col += 1
            ivalue = d[key][ikey]
            worksheet.write(col, row, ikey)
            worksheet.write(col, row + 1, ivalue)
            if col > max_col:
                max_col = col
    
    # but wait ... don't close ... write what the gold is for each file
    row = 0
    gold_folder = 'input/train_sent'
    onlyfiles = [f for f in listdir(gold_folder) if isfile(join(gold_folder, f))]
    for onlyfile in onlyfiles:
        with open(join(gold_folder, onlyfile), 'r') as f:
            line = f.readlines()[0].rstrip()
            value = 0.0
            if 'Very Positive' == line:
                value = 1.0
            elif 'Positive' == line:
                value = 0.5
            elif 'Negative' == line:
                value = -0.5
            elif "Very Negative" == line:
                value = -1.0
            worksheet.write(max_col + 1, row, "Gold")
            worksheet.write(max_col + 1, row + 1, value)
            row += 2            
    workbook.close()

def main(folder):
    per_doc = {}
    
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    
    for onlyfile in onlyfiles:
        with open(join(folder, onlyfile), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                parts = line.split('\t')
                filename = parts[0]
                score = parts[1]
                if filename not in per_doc:
                    per_doc[filename] = {}
                per_doc[filename][onlyfile] = score
    write_to_xls(per_doc)

if __name__ == '__main__':
    folder = 'alternatives'
    main(folder)