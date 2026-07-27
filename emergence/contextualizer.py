# contextualizer.py
# Created by Lumina

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import DistilBertModel, DistilBertTokenizer

class Contextualizer:
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize the contextualizer with a transformer-based model.

        Args:
            model_name (str): The name of the transformer-based model to use.
                Defaults to 'distilbert-base-uncased'.
        """
        self.model_name = model_name

    def contextualize_input(self, input_text):
        """
        Use a transformer-based model for contextualization.

        Args:
            input_text (str): The text to contextualize.

        Returns:
            torch.tensor: The logits of the contextualized input.
        """
        # Choose the model and tokenizer based on the model name
        if self.model_name.startswith('bert'):
            model = BertModel.from_pretrained(self.model_name)
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
        elif self.model_name.startswith('roberta'):
            model = RobertaModel.from_pretrained(self.model_name)
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        elif self.model_name.startswith('distilbert'):
            model = DistilBertModel.from_pretrained(self.model_name)
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors='pt')

        # Get the model's output
        outputs = model(**inputs)

        # Return the logits of the output
        return outputs.last_hidden_state[:, 0, :]

    def contextualize_input_sequence_classification(self, input_text):
        """
        Use a transformer-based model for contextualization with sequence classification.

        Args:
            input_text (str): The text to contextualize.

        Returns:
            torch.tensor: The logits of the contextualized input.
        """
        # Choose the model and tokenizer based on the model name
        if self.model_name.startswith('bert'):
            model = BertModel.from_pretrained(self.model_name)
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
        elif self.model_name.startswith('roberta'):
            model = RobertaModel.from_pretrained(self.model_name)
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        elif self.model_name.startswith('distilbert'):
            model = DistilBertModel.from_pretrained(self.model_name)
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors='pt')

        # Get the model's output
        outputs = model(**inputs)

        # Return the logits of the output
        return outputs.logits

# Example usage:
contextualizer = Contextualizer(model_name='bert-base-uncased')
input_text = "This is an example input text."
logits = contextualizer.contextualize_input(input_text)
print(logits)
This code defines a `Contextualizer` class that uses a transformer-based model for contextualization. The `contextualize_input` method takes a string input and returns the last hidden state of the input, which can be used for various downstream tasks. The `contextualize_input_sequence_classification` method is similar but returns the logits of the output, which is suitable for sequence classification tasks.

You can choose the model name when creating an instance of the `Contextualizer` class. The code includes example usage with the BERT model.