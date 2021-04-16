import training_params
import torch


class PunctuationDataset:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        labels = self.labels[item]

        ids = []
        target_labels = []

        for i, s in enumerate(text):
            inputs = training_params.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )

            input_len = len(inputs)
            ids.extend(inputs)
            target_labels.extend([labels[i]] * input_len)

        ids = ids[:training_params.MAX_LEN - 2]
        target_labels = target_labels[:training_params.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_labels = [0] + target_labels + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = training_params.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_labels = target_labels + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_labels, dtype=torch.long),
        }