from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
model = AutoModel.from_pretrained('ai4bharat/indic-bert')

text = "आपका स्वागत हैं"
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
out = model(input_ids)[0]
print(out.shape)
print(tokenizer.encode(text))