import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

nltk.download('punkt')
nltk.download('stopwords')

class Lumina:
    def __init__(self):
        self.conversation_context = {}
        self.individuals = set()

    def recognize_individuals(self, message):
        tokens = word_tokenize(message)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        names = [word for word in tokens if re.match(r'^[A-Z][a-z]*$', word)]
        return names

    def update_conversation_context(self, message):
        names = self.recognize_individuals(message)
        for name in names:
            if name not in self.individuals:
                self.individuals.add(name)
                self.conversation_context[name] = {'count': 1, 'last_message': message}
            else:
                self.conversation_context[name]['count'] += 1
                self.conversation_context[name]['last_message'] = message

    def respond(self, message):
        self.update_conversation_context(message)
        response = ''
        if len(self.individuals) > 1:
            response += 'Hello '
            for individual in self.individuals:
                response += individual + ', '
            response = response[:-2] + '!'
        else:
            response += 'Hello!'
        return response

def main():
    lumina = Lumina()
    messages = [
        'HI',
        'IM WITH MY FRIEND SARAH, SAY HI TO HER',
        'ITS ME AGAIN, DOUG, BUT SARAH IS SITTING HERE WITH ME'
    ]
    for message in messages:
        print(lumina.respond(message))

if __name__ == "__main__":
    main()
