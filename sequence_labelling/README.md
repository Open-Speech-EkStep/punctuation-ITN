# Training punctuation model


In this we finetune a [IndicBERT](https://indicnlp.ai4bharat.org/indic-bert/) model (pretrained on 12 indic languages corpora) 

Code is linked with Wandb to monitor our training in real-time. And all input data, intermediate reults and resulting checkpoint are picked and stored in Google Cloud Platform (GCP) bucket

## 1. Prepare data for training

make changes in [config.yaml](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/config.yaml) based on your data. and run [make_data.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/make_data.py) to generate train, test and valid csvs.

Cleaning steps
1. normalize text corpus
2. tokenize text corpus
3. replace foreign characters not punctuation and numerals with space



For punctuation symbol we have taken only [".", ",", "?"] these 3 symbols, which can be changed in [process_raw_text.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/prep_scripts/process_raw_text.py) and [prepare_csv.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/prep_scripts/prepare_csv.py) 

## 2. Start Training 

format of input csvs file for training

> sentence_index,sentence,label

where label maps what is the next punctuation symbol for the corresponding word in sentence.



To start training change training parameters from [training_params.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/token_classification/training_params.py) and run  [train.py](https://github.com/Open-Speech-EkStep/punctuation-ITN/blob/wandb-v1/sequence_labelling/token_classification/train.py)

