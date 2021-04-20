import torch
from transformers import AutoTokenizer

TRAIN_DATA = '../input/train.csv'
VALID_DATA = '../input/valid.csv'
MAX_LEN = 128
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
FULL_FINETUNING = True
EPOCHS = 3
MAX_GRAD_NORM = 1.0
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'runs'
