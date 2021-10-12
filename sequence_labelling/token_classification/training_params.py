import torch
from transformers import AutoTokenizer

MAX_LEN = 128
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
FULL_FINETUNING = True
EPOCHS = 5
MAX_GRAD_NORM = 1.0
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'runs'
LOAD_CHECKPOINT = False
CHECKPOINT_PATH = 'checkpoints/checkpoint_last.pt'
LEARNING_RATE = 3e-5
