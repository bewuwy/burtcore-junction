from transformers import AutoTokenizer, AutoModelForSequenceClassification
### from models.py
from bert.models import *
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
model = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
inputs = tokenizer('He is a great guy', return_tensors="pt")
prediction_logits, _ = model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
