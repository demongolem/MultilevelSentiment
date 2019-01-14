import Testbed_Counter

def algorithm_one(output_file):
    # 1) find max count among all the different results files
    count_dicts = Testbed_Counter.fetch_counts()
    max_count = count_dicts[max(count_dicts)]
    # 2) grab lines from each file
    all_entries = Testbed_Counter.fetch_lines()
    # 3) write header
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('filename,method,time,polarity,positive_terms,negative_terms\n')
        # 4) iterate max_count mount of times to write all the rows
        for i in range(0, max_count):
            for all_entry in all_entries:
                if len(all_entries[all_entry]) <= i:
                    continue
                f.write(all_entries[all_entry][i]['test_file'] + ',' 
                        +  str(all_entries[all_entry][i]['method']) + ',' 
                        +  str(all_entries[all_entry][i]['time']) + ',' 
                        +  str(all_entries[all_entry][i]['polarity']) + ','
                        +  str(all_entries[all_entry][i]['positives']) + ','
                        +  str(all_entries[all_entry][i]['negatives']) + ','                                                
                        + '\n') 

def main():
    of = 'master_test_results.csv'
    algorithm_one(of)

if __name__ == '__main__':
    main()