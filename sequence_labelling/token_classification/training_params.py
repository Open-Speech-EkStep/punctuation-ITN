import torch
from transformers import AutoTokenizer

MAX_LEN = 128
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
