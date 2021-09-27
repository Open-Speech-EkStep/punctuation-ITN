import re
import pandas as pd
from tqdm import tqdm
import re
import sys

txt_file_path = sys.argv[1]
csv_path = sys.argv[2]

lines = open(txt_file_path).read().splitlines()



viram_count = []
comma_count = []
qm_count = []

def find_punctuation_count(line):
    punc = re.findall('[.,?]+', line)
    viram_count.append(punc.count('.'))
    comma_count.append(punc.count(','))
    qm_count.append(punc.count('?'))

for line in tqdm(lines):
    find_punctuation_count(line)

df = pd.DataFrame({'sentence': lines, 'viram_count': viram_count, 'comma_count': comma_count, 'qm_count': qm_count})

df.to_csv(csv_path, index=False)
