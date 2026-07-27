# contextualizer.py
# Created by Lumina

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import DistilBertTokenizer, DistilBertModel

class Contextualizer:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = self.get_tokenizer(model_name)
        self.model = self.get_model(model_name)

    def get_tokenizer(self, model_name):
        if 'bert' in model_name:
            return BertTokenizer.from_pretrained(model_name)
        elif 'roberta' in model_name:
            return RobertaTokenizer.from_pretrained(model_name)
        elif 'distilbert' in model_name:
            return DistilBertTokenizer.from_pretrained(model_name)
        else:
            return AutoTokenizer.from_pretrained(model_name)

    def get_model(self, model_name):
        if 'bert' in model_name:
            return BertModel.from_pretrained(model_name)
        elif 'roberta' in model_name:
            return RobertaModel.from_pretrained(model_name)
        elif 'distilbert' in model_name:
            return DistilBertModel.from_pretrained(model_name)
        else:
            return AutoModelForSequenceClassification.from_pretrained(model_name)

    def contextualize_input(self, input_text):
        # Use a transformer-based model for contextualization
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

# Example usage
if __name__ == "__main__":
    contextualizer = Contextualizer(model_name='bert-base-uncased')
    input_text = "This is a sample input text."
    outputs = contextualizer.contextualize_input(input_text)
    print(outputs.shape)
This code defines a `Contextualizer` class that can be used to contextualize input text using various transformer-based models. The `contextualize_input` method takes in input text, tokenizes it, and passes it through the model to obtain the contextualized representation. The `get_tokenizer` and `get_model` methods are used to retrieve the tokenizer and model from the Hugging Face Transformers library based on the specified model name.

You can replace `'bert-base-uncased'` with any other model name from the Hugging Face Transformers library to use a different model for contextualization.