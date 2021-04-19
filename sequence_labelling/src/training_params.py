from transformers import AutoModel, AutoTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 8
EPOCHS = 10
MODEL_PATH = 'model_path'
TRAINING_FILE = '../input/train.csv'
TOKENIZER = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
MODEL = AutoModel.from_pretrained('ai4bharat/indic-bert', return_dict=False)