import os
import Testbed_Bean

def fetch_counts():
    directory_in_str = 'test_results'
    directory = os.fsencode(directory_in_str)

    to_return = {}

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        path = os.path.join(directory_in_str, filename)
        with open(path, 'r') as f:
            lines = f.readlines()
            unique = len(lines) // 2
            to_return[filename] = unique

    return to_return

def fetch_lines():
    directory_in_str = 'test_results'
    directory = os.fsencode(directory_in_str)

    to_return = {}

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        path = os.path.join(directory_in_str, filename)
        with open(path, 'r') as f:
            lines = f.readlines()
            all_entries = []
            for i in range(0, len(lines), 2):
                entry = Testbed_Bean.decode_line_pair(lines[i:i+2])
                all_entries.append(entry)
            to_return[filename] = all_entries

    return to_return

def main():
    lines_dict = fetch_lines()
    print(lines_dict)
    count_dict = fetch_counts()
    print(count_dict)
    

if __name__ == '__main__':
    main()