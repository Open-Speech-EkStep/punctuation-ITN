import torch
from transformers import AlbertForTokenClassification, AlbertTokenizer
import numpy as np
import inference_params
import json
import torch.nn as nn
from indicnlp.tokenize import indic_tokenize

label_encoder_path = inference_params.LABEL_ENCODER_PATH
punctuation_dict = {'hyp': '-', 'qm': '? ', 'comma': ', ', 'end': '। ', 'blank': ' ', 'ex': '! '}

with open(label_encoder_path) as label_encoder:
    train_encoder = json.load(label_encoder)

tokenizer = AlbertTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AlbertForTokenClassification.from_pretrained('ai4bharat/indic-bert',
                                                     num_labels=len(train_encoder),
                                                     output_attentions=False,
                                                     output_hidden_states=False)

model = nn.DataParallel(model)
checkpoint = torch.load(inference_params.CHECKPOINT_PATH)
model.load_state_dict(checkpoint['state_dict'])

model.eval()
model.cuda()


def get_tokens_and_labels_indices_from_text(text):
    tokenized_sentence = tokenizer.encode(text)
    input_ids = torch.tensor([tokenized_sentence]).cuda()
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    return tokens, label_indices


def map_tokens_and_labels_to_word_and_punctuations(text):
    tokens, label_indices = get_tokens_and_labels_indices_from_text(text)
    new_tokens = []
    new_labels = []
    for i in range(1, len(tokens) - 1):
        if tokens[i].startswith("▁"):
            current_word = tokens[i][1:]
            new_labels.append(list(train_encoder.keys())[list(train_encoder.values()).index(label_indices[0][i])])
            for j in range(i + 1, len(tokens) - 1):
                if not tokens[j].startswith("▁"):
                    current_word = current_word + tokens[j]
                if tokens[j].startswith("▁"):
                    break
            new_tokens.append(current_word)
    full_text = ''
    tokenized_text = indic_tokenize.trivial_tokenize_indic(text)
    
    if len(tokenized_text) == len(new_labels):
        full_text_tokens = tokenized_text
    else:
        full_text_tokens = new_tokens
        
    for word, punctuation in zip(full_text_tokens, new_labels):
            full_text = full_text + word + punctuation_dict[punctuation]
    return full_text


if __name__ == "__main__":
    print(map_tokens_and_labels_to_word_and_punctuations("अमेरिका समेत अन्य देशों से जो मदद भारत पहुंची उसका क्या हुआ"))









