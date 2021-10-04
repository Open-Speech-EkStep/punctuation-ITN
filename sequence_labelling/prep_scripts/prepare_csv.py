## usage -> python prepare_csv.py path/to/inp/file/containing/sentences path/to/output/csv

import pandas as pd
import sys
from tqdm import tqdm
import timeit
from joblib import Parallel, delayed

inppath = sys.argv[1] 
outpath = sys.argv[2]


starttime = timeit.default_timer()
df = pd.read_csv(inppath)
print("Time taken to load csv -> ", timeit.default_timer() - starttime)
    
line_list = df['sentence'].to_list()

def fix_for_first_ch_punc(line):
    if line and (line[0] in ['.', ',', '?']):
        line = line[1:]
        return ' '.join(line.split())
    return line

def split_sen_with_label(line):
    line = fix_for_first_ch_punc(line)
    words, labels = [], []
    word_list = line.split()
    for w in word_list:
        if w in [',', '.', '?']:
            if w == ',':
                lab = 'comma'
            elif w == '.':
                lab = 'end'
            elif w == '?':
                lab = 'qm'
            labels.pop()
            labels.append(lab)
        else:
            lab = 'blank'
            words.append(w)
            labels.append(lab)
            
    yield words, labels

outfile = open(outpath, 'w')
print('sentence_index,sentence,label', file=outfile)
def process(ix, line):
    g = split_sen_with_label(line)
    words, labels = next(g)
    if len(words) == len(labels):
        return [ix+1," ".join(words)," ".join(labels)]

starttime = timeit.default_timer()

out = Parallel(n_jobs=-1)(delayed(process)(ix, line) for ix, line in tqdm(enumerate(line_list)))

for i in tqdm(out):
    print(*i, sep = ',', file=outfile)

print("Time taken -> ", timeit.default_timer() - starttime)

outfile.close()
