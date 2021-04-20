import datetime
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AlbertForTokenClassification, AdamW, get_linear_schedule_with_warmup

import training_params
from dataset_loader import PunctuationDataset

warnings.filterwarnings('ignore')


def process_data(data_csv):
    df = pd.read_csv(data_csv)
    sentences = df.groupby("sentence")["word"].apply(list).values
    labels = df.groupby("sentence")["label"].apply(list).values
    tag_values = list(set(df["label"].values))
    tag_values.append("PAD")
    encoder = {t: i for i, t in enumerate(tag_values)}
    return sentences, labels, encoder, tag_values


folder_hook = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_folder = training_params.LOG_DIR + '/' + folder_hook
checkpoint_folder = training_params.CHECKPOINT_DIR + '/' + folder_hook

os.makedirs(log_folder, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)

writer = SummaryWriter(log_folder)

train_sentences, train_labels, train_encoder, tag_values = process_data(training_params.TRAIN_DATA)
valid_sentences, valid_labels, _, _ = process_data(training_params.VALID_DATA)

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
model.to(training_params.DEVICE)

step_count = 0
for epoch in range(training_params.EPOCHS):

    model.train()
    total_loss = 0

    # Training loop
    tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)), unit='batch')
    tk0.set_description(f'Epoch {epoch + 1}')

    for step, batch in enumerate(tk0):
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

        # loss for step
        writer.add_scalar("Training Loss- Step", loss, step_count)
        step_count+=1

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=training_params.MAX_GRAD_NORM)

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_data_loader)
    print("Average train loss: {}".format(avg_train_loss))
    writer.add_scalar("Training Loss", avg_train_loss, epoch)

    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_folder + '/checkpoint_last.pt')
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []

    best_val_loss = np.inf
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

    if eval_loss < best_val_loss:
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, checkpoint_folder + '/checkpoint_best.pt')
        best_val_loss = eval_loss

    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    writer.add_scalar("Validation Loss", eval_loss, epoch)

    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels) for p_i, l_i in zip(p, l) if
                 tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels for l_i in l if tag_values[l_i] != "PAD"]

    val_accuracy = accuracy_score(pred_tags, valid_tags)
    val_f1_score = f1_score(pred_tags, valid_tags, average='macro')
    print("Validation Accuracy: {}".format(val_accuracy))
    print("Validation F1-Score: {}".format(val_f1_score))
    print("Classification Report: {}".format(classification_report(pred_tags, valid_tags, output_dict=True,
                                                                   labels=np.unique(pred_tags))))
    writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
    writer.add_scalar('Validation F1 score', val_f1_score, epoch)
