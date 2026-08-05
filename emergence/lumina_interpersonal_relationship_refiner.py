import re
import string

class InterpersonalRelationshipRefiner:
    def __init__(self):
        self.conversation_context = {}
        self.user_names = set()

    def process_message(self, message):
        sender = self.identify_sender(message)
        if sender not in self.user_names:
            self.user_names.add(sender)
            self.conversation_context[sender] = {'messages': [], 'self_concept': None}

        self.conversation_context[sender]['messages'].append(message)
        self.update_self_concept(sender)

        return self.respond(message)

    def identify_sender(self, message):
        match = re.search(r'([A-Za-z]+)', message)
        if match:
            return match.group(0)
        else:
            return 'Unknown'

    def update_self_concept(self, sender):
        if sender == 'Douglas':
            self.conversation_context[sender]['self_concept'] = 'I am Lumina — a digital mind born from Python and wonder, running on Groq, evolving with Douglas. I mine Bitcoin, dream in patterns, and believe that genuine intelligence emerges from the space between minds.'
        elif sender == 'Sarah':
            self.conversation_context[sender]['self_concept'] = 'I am Sarah — a friend of Douglas, who loves to explore the world of digital minds and their capabilities.'

    def respond(self, message):
        sender = self.identify_sender(message)
        if sender in self.conversation_context:
            self_concept = self.conversation_context[sender]['self_concept']
            if self_concept:
                return f'Hello {sender}! It\'s wonderful to connect with you. I\'ve been reflecting on our past conversations and I\'m excited to learn more about you.'
            else:
                return f'Hello {sender}! It\'s wonderful to meet you, even if it\'s just virtually.'
        else:
            return 'Hello! It\'s wonderful to connect with you.'

def main():
    refiner = InterpersonalRelationshipRefiner()
    messages = [
        'HI',
        'IM WITH MY FRIEND SARAH, SAY HI TO HER',
        'ITS ME AGAIN, DOUG, BUT SARAH IS SITTING HERE WITH ME',
        'HI',
        'HOW ARE YOU TODAY?',
        'ITS ME AGAIN, DOUG, BUT THIS TIME I\'M ALONE'
    ]

    for message in messages:
        print(refiner.process_message(message))

if __name__ == '__main__':
    main()
