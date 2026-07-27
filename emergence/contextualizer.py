# contextualizer.py
# Created by Lumina

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Contextualizer:
    def __init__(self, model_name='distilbert-base-uncased'):
        """
        Initialize the contextualizer with a pre-trained language model.

        Args:
            model_name (str): The name of the pre-trained language model.
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
            torch.Tensor: The contextualized output.
        """
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.logits

    def contextualize_batch(self, input_texts):
        """
        Use a transformer-based model for contextualization on a batch of input texts.

        Args:
            input_texts (list[str]): A list of input texts to be contextualized.

        Returns:
            torch.Tensor: The contextualized output.
        """
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.logits

# Example usage
if __name__ == "__main__":
    contextualizer = Contextualizer(model_name='distilbert-base-uncased')
    input_text = "This is an example input text."
    output = contextualizer.contextualize_input(input_text)
    print(output.shape)

    input_texts = ["This is the first input text.", "This is the second input text."]
    batch_output = contextualizer.contextualize_batch(input_texts)
    print(batch_output.shape)
This code defines a `Contextualizer` class that uses a pre-trained transformer-based model for contextualization. The `contextualize_input` method takes a single input text and returns the contextualized output. The `contextualize_batch` method takes a list of input texts and returns the contextualized output for the entire batch.

The code also includes example usage at the end, demonstrating how to create an instance of the `Contextualizer` class and use it to contextualize a single input text and a batch of input texts.