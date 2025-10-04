# finetuned from irlab-udc/MetaHateBERT
# with dataset irlab-udc/metahate

from transformers import pipeline

# Load the model
classifier = pipeline("text-classification", model="irlab-udc/MetaHateBERT")

# Test the model
while True:
    input_text = input()
    result = classifier(input_text)
    print(result)