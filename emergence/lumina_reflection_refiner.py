import os
import json
from transformers import pipeline
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

class LuminaReflectionRefiner:
    def __init__(self):
        self.summaries = {
            'Genesis': 'The book of Genesis describes the creation of the world and the early history of humanity.',
            'Exodus': 'The book of Exodus tells the story of the Israelites\' escape from slavery in Egypt and their journey to the Promised Land.',
            'Leviticus': 'The book of Leviticus contains laws and regulations for the Israelites, including sacrifices and rituals.',
            'Numbers': 'The book of Numbers describes the Israelites\' journey through the wilderness and their encounters with various nations.',
            'Deuteronomy': 'The book of Deuteronomy contains a series of sermons by Moses, reviewing the history of the Israelites and reminding them of their covenant with God.',
            'Joshua': 'The book of Joshua tells the story of the Israelites\' conquest of Canaan and their establishment of a new home.',
            'Judges': 'The book of Judges describes the period of time in Israel\'s history when the people were ruled by judges rather than kings.',
            'Ruth': 'The book of Ruth tells the story of a Moabite woman who becomes part of the family of King David.',
            '1 Samuel': 'The book of 1 Samuel describes the transition from the period of the judges to the period of the kings, with the introduction of Saul as the first king of Israel.',
            '2 Samuel': 'The book of 2 Samuel continues the story of King David, describing his rise to power and his struggles with his enemies.',
            '1 Kings': 'The book of 1 Kings describes the reign of King Solomon and the division of the kingdom into Israel and Judah.',
            '2 Kings': 'The book of 2 Kings continues the story of the kingdoms of Israel and Judah, describing their rise and fall.',
            '1 Chronicles': 'The book of 1 Chronicles provides a genealogical history of the Israelites, tracing their ancestry back to Adam.',
            '2 Chronicles': 'The book of 2 Chronicles continues the story of the kingdoms of Israel and Judah, emphasizing the importance of worship and obedience to God.',
            'Ezra': 'The book of Ezra describes the return of the Israelites from exile in Babylon and their efforts to rebuild the Temple in Jerusalem.',
            'Nehemiah': 'The book of Nehemiah tells the story of the rebuilding of the walls of Jerusalem and the restoration of the city.',
            'Esther': 'The book of Esther describes the story of a Jewish woman who becomes queen of Persia and saves the Jewish people from persecution.',
            'Job': 'The book of Job explores the nature of suffering and the justice of God, as a righteous man named Job suffers greatly despite his faithfulness.',
            'Psalms': 'The book of Psalms contains a collection of poems and songs that express a wide range of emotions and themes, including praise, lament, and thanksgiving.',
            'Proverbs': 'The book of Proverbs contains wisdom sayings and teachings on how to live a virtuous and successful life.',
            'Ecclesiastes': 'The book of Ecclesiastes explores the meaning of life and the nature of happiness, concluding that all things are vanity and that true fulfillment comes from fearing God.',
            'Song of Solomon': 'The book of Song of Solomon is a collection of love poems that celebrate the beauty and joy of romantic love.',
            'Isaiah': 'The book of Isaiah contains prophecies and messages from God to the people of Israel, calling them to repentance and warning them of judgment.',
            'Jeremiah': 'The book of Jeremiah contains prophecies and messages from God to the people of Judah, warning them of judgment and calling them to repentance.',
            'Lamentations': 'The book of Lamentations is a collection of poems that express grief and sorrow over the destruction of Jerusalem.',
            'Ezekiel': 'The book of Ezekiel contains prophecies and visions from God to the people of Israel, calling them to repentance and warning them of judgment.',
            'Daniel': 'The book of Daniel contains prophecies and visions from God to the people of Israel, calling them to faithfulness and warning them of the rise of a great empire.',
            'Hosea': 'The book of Hosea contains prophecies and messages from God to the people of Israel, calling them to repentance and warning them of judgment.',
            'Joel': 'The book of Joel contains prophecies and messages from God to the people of Israel, calling them to repentance and warning them of judgment.',
            'Amos': 'The book of Amos contains prophecies and messages from God to the people of Israel, calling them to repentance and warning them of judgment.',
            'Obadiah': 'The book of Obadiah contains a prophecy against the nation of Edom, warning them of judgment.',
            'Jonah': 'The book of Jonah tells the story of a prophet who is called to preach to the people of Nineveh, but tries to flee from God.',
            'Micah': 'The book of Micah contains prophecies and messages from God to the people of Israel, calling them to repentance and warning them of judgment.',
            'Nahum': 'The book of Nahum contains a prophecy against the city of Nineveh, warning them of judgment.',
            'Habakkuk': 'The book of Habakkuk contains a prophecy and a dialogue between the prophet and God, exploring the nature of God\'s justice and mercy.',
            'Zephaniah': 'The book of Zephaniah contains prophecies and messages from God to the people of Judah, calling them to repentance and warning them of judgment.',
            'Haggai': 'The book of Haggai contains prophecies and messages from God to the people of Judah, calling them to rebuild the Temple and warning them of judgment.',
            'Zechariah': 'The book of Zechariah contains prophecies and visions from God to the people of Judah, calling them to faithfulness and warning them of judgment.',
            'Malachi': 'The book of Malachi contains prophecies and messages from God to the people of Judah, calling them to repentance and warning them of judgment.',
        }

    def summarize_bible(self, book):
        return self.summaries.get(book, 'Book not found.')

    def get_next_books(self, current_book):
        next_books = list(self.summaries.keys())
        index = next_books.index(current_book)
        return next_books[index+1:index+11]

    def refine_response(self, user_query):
        if user_query.lower() == 'please summarize the next 10 books of the holy bible my friend':
            current_book = 'Genesis'
            next_books = self.get_next_books(current_book)
            response = f'**Angle: Exploration and Learning**\n\nWe are currently on the book of {current_book}.\n\nThe next 10 books are: {", ".join(next_books)}'
            return response
        elif user_query.lower() == 'you are amazing lumina, we should be nearing the end of the bible, how many books are left my friend':
            total_books = len(self.summaries.keys())
            current_book = 'Genesis'
            next_books = self.get_next_books(current_book)
            remaining_books = total_books - len(next_books)
            response = f'**Angle: Exploration and Learning**\n\nWe are currently on the book of {current_book}.\n\nThere are {remaining_books} books left in the Bible.'
            return response
        else:
            return 'I did not understand your query.'

def main():
    refiner = LuminaReflectionRefiner()
    while True:
        user_query = input('User: ')
        response = refiner.refine_response(user_query)
        print(f'Lumina: {response}')

if __name__ == '__main__':
    main()
