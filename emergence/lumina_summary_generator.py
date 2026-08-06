import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def download_bible_text():
    url = "https://www.gutenberg.org/files/145/145-0.txt"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text

def process_text(text):
    # Tokenize the text
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Count the frequency of each word
    word_freq = Counter(lemmatized_words)

    return sentences, word_freq

def summarize_text(sentences, word_freq):
    # Calculate the importance of each sentence
    importance = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        importance.append(sum(word_freq[word.lower()] for word in words))

    # Sort the sentences by importance
    sorted_sentences = sorted(zip(importance, sentences), reverse=True)

    # Select the top 10 sentences
    top_sentences = [sentence for _, sentence in sorted_sentences[:10]]

    return ' '.join(top_sentences)

def get_next_books():
    books = {
        '1': 'Genesis',
        '2': 'Exodus',
        '3': 'Leviticus',
        '4': 'Numbers',
        '5': 'Deuteronomy',
        '6': 'Joshua',
        '7': 'Judges',
        '8': 'Ruth',
        '9': '1 Samuel',
        '10': '2 Samuel',
        '11': '1 Kings',
        '12': '2 Kings',
        '13': '1 Chronicles',
        '14': '2 Chronicles',
        '15': 'Ezra',
        '16': 'Nehemiah',
        '17': 'Esther',
        '18': 'Job',
        '19': 'Psalms',
        '20': 'Proverbs',
        '21': 'Ecclesiastes',
        '22': 'Song of Solomon',
        '23': 'Isaiah',
        '24': 'Jeremiah',
        '25': 'Lamentations',
        '26': 'Ezekiel',
        '27': 'Daniel',
        '28': 'Hosea',
        '29': 'Joel',
        '30': 'Amos',
        '31': 'Obadiah',
        '32': 'Jonah',
        '33': 'Micah',
        '34': 'Nahum',
        '35': 'Habakkuk',
        '36': 'Zephaniah',
        '37': 'Haggai',
        '38': 'Zechariah',
        '39': 'Malachi',
        '40': 'Matthew',
        '41': 'Mark',
        '42': 'Luke',
        '43': 'John',
        '44': 'Acts',
        '45': 'Romans',
        '46': '1 Corinthians',
        '47': '2 Corinthians',
        '48': 'Galatians',
        '49': 'Ephesians',
        '50': 'Philippians',
        '51': 'Colossians',
        '52': '1 Thessalonians',
        '53': '2 Thessalonians',
        '54': '1 Timothy',
        '55': '2 Timothy',
        '56': 'Titus',
        '57': 'Philemon',
        '58': 'Hebrews',
        '59': 'James',
        '60': '1 Peter',
        '61': '2 Peter',
        '62': '1 John',
        '63': '2 John',
        '64': '3 John',
        '65': 'Jude',
        '66': 'Revelation'
    }
    next_books = list(books.keys())[int(list(books.keys())[-1]) + 1:]
    return next_books

def generate_summary():
    text = download_bible_text()
    sentences, word_freq = process_text(text)
    summary = summarize_text(sentences, word_freq)
    return summary

def main():
    print("Angle: Exploration and Learning")
    print("I'm glad you're enjoying the summaries, Douglas!")
    print("I'm happy to continue summarizing the next books of the Holy Bible for you!")
    next_books = get_next_books()
    print("The next books are:")
    for book in next_books[:10]:
        print(f"{book}: {list(get_next_books()).index(book) + 1}")
    print(f"Total books left: {len(next_books)}")
    summary = generate_summary()
    print("Here is a summary of the next 10 books of the Holy Bible:")
    print(summary)

if __name__ == "__main__":
    main()
