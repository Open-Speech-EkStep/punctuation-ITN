import re
import sys
from tqdm import tqdm

def filter_line(line):
    '''
    replace foreign characters not punctuation and numerals with space
    '''
    
    if re.search(punc,line):     
        clean_line = ' '.join(re.sub(pattern, ' ', line).split())
        if clean_line and (clean_line[0] in ['.', ',', '?']):
            clean_line = clean_line[1:]
        temp_line = clean_line.replace(" ","")
        if regex.search(temp_line):
            return ''
        return ' '.join(clean_line.split())
    return ''

if __name__ == '__main__':
    fpath = sys.argv[1]
    dest = sys.argv[2]
    with open('../data/dict.ltr.txt', mode = 'r', encoding='UTF-8') as file:
        dictionary = file.readlines()
    char = [i.split(' ')[0] for i in dictionary] 
    pattern = '[^ '+''.join(char)+'.,?]'
    punc = '[.,?]+'
    regex = re.compile(r'[.?,]{2,}')
    line_list = open(fpath).read().splitlines()
#     pattern = '[^ ઁ-ઃઅ-ઋઍએ-ઑઓ-નપ-રલ-ળવ-હા-ૅે-ૉો-્.,?]+'
    
    cleaned_line_list = [filter_line(line) for line in tqdm(line_list)]
    cleaned_line_list = [i for i in cleaned_line_list if i != '']  

    with open(dest, 'w') as f:
        f.write('\n'.join(cleaned_line_list))
    print("\tcompleted")

