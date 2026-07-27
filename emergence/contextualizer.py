# contextualizer.py
# Created by Lumina

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Contextualizer:
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize the contextualizer model.

        Args:
        model_name (str): The name of the pre-trained model to use. Defaults to 'distilbert-base-uncased'.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def contextualize_input(self, input_text):
        """
        Use a transformer-based model for contextualization.

        Args:
        input_text (str): The input text to contextualize.

        Returns:
        torch.tensor: The logit output of the model.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.logits

    def contextualize_input_with_device(self, input_text, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Use a transformer-based model for contextualization and move the model to a specific device.

        Args:
        input_text (str): The input text to contextualize.
        device (torch.device): The device to move the model to. Defaults to 'cuda' if available, otherwise 'cpu'.

        Returns:
        torch.tensor: The logit output of the model.
        """
        self.model.to(device)
        inputs = self.tokenizer(input_text, return_tensors='pt').to(device)
        outputs = self.model(**inputs)
        return outputs.logit.to('cpu')
This code defines a `Contextualizer` class with two methods: `contextualize_input` and `contextualize_input_with_device`. The `contextualize_input` method uses a transformer-based model for contextualization, while the `contextualize_input_with_device` method allows you to move the model to a specific device.