import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from training_params import TOKENIZER
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

class PunctuationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tag2idx):
        self.texts = texts
        self.labels = labels
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        sentence = self.texts[item].split()
        text_label = self.labels[item].split()

        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_label):
            # Tokenize the word and count number of subwords
            tokenized_word = TOKENIZER.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        input_ids = pad_sequences([TOKENIZER.convert_tokens_to_ids(tokenized_sentence)],
                                  maxlen=128, dtype="long", value=0.0,
                                  truncating="post", padding="post")

        tags = pad_sequences([[self.tag2idx.get(l) for l in labels]],
                             maxlen=128, value=self.tag2idx["PAD"], padding="post",
                             dtype="long", truncating="post")

        attention_masks = [float(i != 0.0) for i in input_ids[0]]

        return {
            "ids": torch.tensor(input_ids[0], dtype=torch.long),
            "mask": torch.tensor(attention_masks, dtype=torch.long),
            "target_tag": torch.tensor(tags[0], dtype=torch.long),
        }


if __name__=="__main__":
    df = pd.read_csv('data/train_sample.csv')
    print("File Read")
    tag_values = ['blank', 'end', 'comma', 'qm']
    tag_values.append("PAD")
    encoder = {t: i for i, t in enumerate(tag_values)}
    print(encoder)
    '''
    def split_string(line):
        return str(line).split()
    sentences = Parallel(n_jobs=-1)(delayed(split_string)(s) for s in tqdm(df['sentence']))
    sentences = np.asarray(sentences)
    punctuations = Parallel(n_jobs=-1)(delayed(split_string)(s) for s in tqdm(df['label']))
    punctuations = np.asarray(punctuations)
    '''
    sentences = df['sentence'].values
    punctuations = df['label'].values
    d = PunctuationDataset(sentences, punctuations, encoder).__getitem__(0)
    print(type(d))
    print(d['ids'])
    print(d['mask'])
    print(d['target_tag'])