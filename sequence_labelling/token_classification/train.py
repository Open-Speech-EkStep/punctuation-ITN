from dataset_loader import PunctuationDataset
import pandas as pd
import torch
import training_params
from tqdm import trange
from seqeval.metrics import f1_score, accuracy_score
from transformers import AlbertForTokenClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np


def process_data(data_csv):
    df = pd.read_csv(data_csv)
    sentences = df.groupby("sentence")["word"].apply(list).values
    labels = df.groupby("sentence")["label"].apply(list).values
    tag_values = list(set(df["label"].values))
    tag_values.append("PAD")
    encoder = {t: i for i, t in enumerate(tag_values)}
    return sentences, labels, encoder, tag_values


train_sentences, train_labels, train_encoder, tag_values = process_data('../input/train.csv')
valid_sentences, valid_labels, _, _ = process_data('../input/valid.csv')

train_dataset = PunctuationDataset(texts=train_sentences, labels=train_labels,
                                   tag2idx=train_encoder)
valid_dataset = PunctuationDataset(texts=valid_sentences, labels=valid_labels,
                                   tag2idx=train_encoder)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_params.BATCH_SIZE, num_workers=4)
valid_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_params.BATCH_SIZE, num_workers=4)

model = AlbertForTokenClassification.from_pretrained('ai4bharat/indic-bert',
                                                     num_labels=len(train_encoder),
                                                     output_attentions=False,
                                                     output_hidden_states=False)


if training_params.FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

total_steps = len(train_data_loader) * training_params.EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_values, validation_loss_values = [], []

for _ in trange(training_params.EPOCHS, desc="Epoch"):

    model.train()
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_data_loader):
        # add batch to gpu
        for k, v in batch.items():
            batch[k] = v.to(training_params.DEVICE)

        b_input_ids, b_input_mask, b_labels = batch['ids'], batch['mask'], batch['target_tag']

        model.zero_grad()

        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]

        loss.backward()

        total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=training_params.MAX_GRAD_NORM)

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_data_loader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_data_loader:
        for k, v in batch.items():
            batch[k] = v.to(training_params.DEVICE)
        b_input_ids, b_input_mask, b_labels = batch['ids'], batch['mask'], batch['target_tag']

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_data_loader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    #print(predictions)
    #print(true_labels)
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels) for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels for l_i in l if tag_values[l_i] != "PAD"]
    print(pred_tags)
    print(true_labels)
    #print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    #print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    #print()