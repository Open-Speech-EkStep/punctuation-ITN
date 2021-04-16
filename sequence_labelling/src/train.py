import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import training_params
import dataset_loader
import runner
from model import PunctuationModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="utf-8")

    enc_label = preprocessing.LabelEncoder()

    df.loc[:, "label"] = enc_label.fit_transform(df["label"])

    sentences = df.groupby("sentence")["word"].apply(list).values
    labels = df.groupby("sentence")["label"].apply(list).values
    return sentences, labels, enc_label


if __name__ == "__main__":
    sentences, labels, enc_label = process_data(training_params.TRAINING_FILE)

    meta_data = {
        "enc_label": enc_label,
    }

    joblib.dump(meta_data, "meta.bin")

    num_labels = len(list(enc_label.classes_))

    (
        train_sentences,
        test_sentences,
        train_labels,
        test_labels,
    ) = model_selection.train_test_split(sentences, labels, random_state=42, test_size=0.1)

    train_dataset = dataset_loader.PunctuationDataset(
        texts=train_sentences, labels=train_labels,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=training_params.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset_loader.PunctuationDataset(
        texts=test_sentences, labels=test_labels,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=training_params.VALID_BATCH_SIZE, num_workers=1
    )

    # device = torch.device("cuda")
    device = 'cpu'
    model = PunctuationModel(num_tag=num_labels)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / training_params.TRAIN_BATCH_SIZE * training_params.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(training_params.EPOCHS):
        train_loss = runner.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = runner.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), training_params.MODEL_PATH)
            best_loss = test_loss
