# contextualizer.py
# Created by Lumina

import logging
import os
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Contextualizer:
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """
        Initializes the Contextualizer model.

        Args:
        model_name (str): The name of the transformer-based model to use for contextualization.
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """
        Loads the transformer-based model and tokenizer.
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info(f"Model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {str(e)}")

    def contextualize_input(self, input_text: str) -> Optional[float]:
        """
        Contextualizes the input text using the loaded transformer-based model.

        Args:
        input_text (str): The input text to be contextualized.

        Returns:
        Optional[float]: The contextualized output or None if the model failed to load.
        """
        if self.model is None or self.tokenizer is None:
            self._load_model()

        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(input_text, return_tensors='pt')
            outputs = self.model(**inputs)
            return outputs.logits
        else:
            logger.error("Failed to contextualize input: Model not loaded.")
            return None

    def save_model(self, output_dir: str):
        """
        Saves the loaded model and tokenizer to the specified output directory.

        Args:
        output_dir (str): The directory where the model and tokenizer will be saved.
        """
        if self.model is not None and self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            self.model.save_pretrained(output_dir)
            logger.info(f"Model saved to '{output_dir}' successfully.")
        else:
            logger.error("Failed to save model: Model not loaded.")

    def load_model(self, input_dir: str):
        """
        Loads the saved model and tokenizer from the specified input directory.

        Args:
        input_dir (str): The directory where the saved model and tokenizer are located.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(input_dir)
            logger.info(f"Model loaded from '{input_dir}' successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from '{input_dir}': {str(e)}")
This refactored code introduces a modular architecture for the Contextualizer model, allowing for easier maintenance and updates. The model and tokenizer are now loaded on demand, and the `save_model` and `load_model` methods provide a way to persist the model and tokenizer to and from disk.