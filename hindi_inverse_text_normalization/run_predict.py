# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentParser
from typing import List

from inverse_normalize import INVERSE_NORMALIZERS


'''
Runs denormalization prediction on text data
'''


def load_file(file_path: str) -> List[str]:
    """
    Load given text file into list of string.

    Args: 
        file_path: file path

    Returns: flat list of string
    """
    res = []
    with open(file_path, 'r') as fp:
        for line in fp:
            if line:
                res.append(line.strip())
    return res


def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.

    Args:
        file_path: file path
        data: list of string
        
    """
    with open(file_path, 'w') as fp:
        for line in data:
            fp.write(line + '\n')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input file path", required=True, type=str)
    parser.add_argument("--verbose", help="print denormalization info. For debugging", action='store_true')
    parser.add_argument("--inverse_normalizer", default='nemo', type=str)
    parser.add_argument("--output", help="output file path", required=False, type=str)

    return parser.parse_args()

def remove_starting_zeros(word, hindi_digits_with_zero):
    if word[0] in hindi_digits_with_zero and len(word) > 1:
        pos_non_zero_nums = [pos for pos, word in enumerate(list(word)) if word != "०"]
        # print(pos_non_zero_nums, word)
        first_non_zero_num = min(pos_non_zero_nums)
        word = word[first_non_zero_num:]

    return word

if __name__ == "__main__":
    args = parse_args()
    file_path = args.input
    inverse_normalizer = INVERSE_NORMALIZERS[args.inverse_normalizer]

    print("Loading data: " + file_path)
    data = load_file(file_path)
    hindi_digits_with_zero = '०१२३४५६६७८९'

    # print("- Data: " + str(len(data)) + " sentences")
    inverse_normalizer_prediction = inverse_normalizer(data, verbose=False)

    astr_list = []
    print(inverse_normalizer_prediction)
    print('-'*100)
    inverse_normalizer_prediction = [sent.replace('\r', '') for sent in inverse_normalizer_prediction]
    print(inverse_normalizer_prediction)
    for sent in inverse_normalizer_prediction:
        astr_list.append(' '.join([remove_starting_zeros(word, hindi_digits_with_zero) for word in sent.split(' ')]))

    print('Trimmed output')
    print(astr_list)
    # write_file(args.output, inverse_normalizer_prediction)
    # print(f"- Normalized. Writing out to {args.output}")
