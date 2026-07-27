# contextualizer.py
# Created by Lumina

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Contextualizer:
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize the contextualizer with a pre-trained transformer model.

        Args:
        model_name (str): The name of the pre-trained transformer model.
            Defaults to 'distilbert-base-uncased'.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def contextualize_input(self, input_text):
        """
        Use a transformer-based model for contextualization.

        Args:
        input_text (str): The input text to be contextualized.

        Returns:
        torch.Tensor: The logit output of the transformer model.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.logits

    def contextualize_input_batch(self, input_texts):
        """
        Use a transformer-based model for contextualization on a batch of input texts.

        Args:
        input_texts (list[str]): A list of input texts to be contextualized.

        Returns:
        torch.Tensor: The logit output of the transformer model for each input text.
        """
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.logit
This code defines a `Contextualizer` class that uses a pre-trained transformer model for contextualization. The `contextualize_input` method takes a single input text and returns the logit output of the transformer model. The `contextualize_input_batch` method takes a list of input texts and returns the logit output for each input text. The model and tokenizer are initialized in the `__init__` method, and the `contextualize_input` and `contextualize_input_batch` methods are used to contextualize the input texts.