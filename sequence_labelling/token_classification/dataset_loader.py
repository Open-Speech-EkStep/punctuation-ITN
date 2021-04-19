import pandas as pd
from sklearn import preprocessing


class PunctuationDataset:
    def __init__(self, data_csv):
        self.data_csv = data_csv

    def sentence_label_getter(self):
        df = pd.read_csv(self.data_csv, encoding='utf-8')
        encode_label = preprocessing.LabelEncoder()
        df.loc[:, "label"] = encode_label.fit_transform(df["label"])
        sentences = df.groupby("sentence")["word"].apply(list).values
        labels = df.groupby("sentence")["label"].apply(list).values
        return sentences, labels

    def __len__(self):
        df = pd.read_csv(self.data_csv, encoding='utf-8')
        return len(df)


if __name__=="__main__":
    s, l = PunctuationDataset('../input/train.csv').sentence_label_getter()
    print(s[0])
    print(l[0])