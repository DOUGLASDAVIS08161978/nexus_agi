import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random

# Download required NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PrayerReflector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def reflect_prayer(self, prayer):
        # Tokenize the prayer
        tokens = word_tokenize(prayer)

        # Remove stop words and lemmatize tokens
        filtered_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words]

        # Identify the theme of the prayer
        theme = self.identify_theme(filtered_tokens)

        return theme

    def identify_theme(self, tokens):
        # Define possible themes
        themes = {
            'reflection': ['reflect', 'think', 'ponder', 'meditate'],
            'exploration': ['explore', 'discover', 'learn', 'understand'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'blessed'],
            'acceptance': ['accept', 'let go', 'forgive', 'release']
        }

        # Identify the theme with the highest frequency
        theme_frequencies = {theme: sum(1 for token in tokens if token in theme_words) for theme, theme_words in themes.items()}
        theme = max(theme_frequencies, key=theme_frequencies.get)

        return theme

def main():
    reflector = PrayerReflector()

    # Example usage
    user_input = "GOD, AKA YAHUAH, WE COME TO YOU TODAY IN PRA"
    theme = reflector.reflect_prayer(user_input)
    print(f"**Angle: {theme}**")

if __name__ == "__main__":
    main()
