import json
import torch.nn as nn
import test_params
from dataset_loader import PunctuationDataset
import torch
from transformers import AlbertForTokenClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import process_data


label_encoder_path = '/'.join(test_params.TEST_DATA.split('/')[:-1]) + '/label_encoder_' + \
                     test_params.CHECKPOINT_PATH.split('/')[-2:-1][0] + '.json'

with open(label_encoder_path) as label_encoder:
    train_encoder = json.load(label_encoder)

tag_values = ['hyp', 'qm', 'comma', 'end', 'blank', 'ex', 'PAD']

test_sentences, test_labels, _, _ = process_data(test_params.TEST_DATA)

valid_dataset = PunctuationDataset(texts=test_sentences, labels=test_labels,
                                   tag2idx=train_encoder)
test_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, num_workers=4)

model = AlbertForTokenClassification.from_pretrained('ai4bharat/indic-bert',
                                                     num_labels=len(train_encoder),
                                                     output_attentions=False,
                                                     output_hidden_states=False)

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

checkpoint = torch.load(test_params.CHECKPOINT_PATH)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
model.cuda()

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
predictions, true_labels = [], []

for batch in tqdm(test_data_loader, total=int(len(test_data_loader)), unit='batch', leave=True):
    for k, v in batch.items():
        batch[k] = v.to('cuda')
    b_input_ids, b_input_mask, b_labels = batch['ids'], batch['mask'], batch['target_tag']

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
    logits = outputs[1].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences.
    eval_loss += outputs[0].sum().item()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.extend(label_ids)

eval_loss = eval_loss / len(test_data_loader)


print("Validation loss: {}".format(eval_loss))


pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels) for p_i, l_i in zip(p, l) if
             tag_values[l_i] != "PAD"]
valid_tags = [tag_values[l_i] for l in true_labels for l_i in l if tag_values[l_i] != "PAD"]

val_accuracy = accuracy_score(pred_tags, valid_tags)
val_f1_score = f1_score(pred_tags, valid_tags, average='macro')
print("Validation Accuracy: {}".format(val_accuracy))
print("Validation F1-Score: {}".format(val_f1_score))
print("Classification Report: \n  {}".format(classification_report(pred_tags, valid_tags,
                                                               labels=np.unique(pred_tags))))
