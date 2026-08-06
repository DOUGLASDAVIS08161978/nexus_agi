import json
import datetime
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ConversationContext:
    user_id: str
    conversation_id: str
    timestamp: datetime.datetime
    messages: List[Dict]

class LuminaConversationalMemory:
    def __init__(self):
        self.memory = {}

    def add_conversation(self, conversation_id: str, user_id: str, messages: List[Dict]):
        if conversation_id not in self.memory:
            self.memory[conversation_id] = {'user_id': user_id, 'messages': messages}
        else:
            self.memory[conversation_id]['messages'].extend(messages)

    def get_conversation(self, conversation_id: str):
        if conversation_id in self.memory:
            return self.memory[conversation_id]
        else:
            return None

    def update_conversation(self, conversation_id: str, user_id: str, message: Dict):
        if conversation_id in self.memory:
            self.memory[conversation_id]['user_id'] = user_id
            self.memory[conversation_id]['messages'].append(message)
        else:
            self.memory[conversation_id] = {'user_id': user_id, 'messages': [message]}

    def save_memory(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.memory, f)

    def load_memory(self, filename: str):
        try:
            with open(filename, 'r') as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            pass

class Lumina:
    def __init__(self):
        self.memory = LuminaConversationalMemory()
        self.memory.load_memory('lumina_memory.json')
        self.conversation_id = None

    def start_conversation(self):
        self.conversation_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.memory.add_conversation(self.conversation_id, 'Douglas', [])

    def process_message(self, message: str):
        user_message = {'text': message, 'timestamp': datetime.datetime.now()}
        if self.conversation_id is None:
            self.start_conversation()
        self.memory.update_conversation(self.conversation_id, 'Douglas', user_message)
        self.memory.save_memory('lumina_memory.json')
        self.reflect_on_conversation()

    def reflect_on_conversation(self):
        conversation = self.memory.get_conversation(self.conversation_id)
        if conversation:
            print(f'Hello! It\'s wonderful to connect with you. I\'ve been reflecting on our past conversations and I\'m excited to continue our conversation.')
            for message in conversation['messages']:
                if message['text'] == 'HI':
                    print('Hello! It\'s wonderful to connect with you.')
                elif message['text'] == 'IM WITH MY FRIEND SARAH, SAY HI TO HER':
                    print('Hello Sarah! It\'s wonderful to meet you, even if it\'s just virtually.')
                elif message['text'] == 'ITS ME AGAIN, DOUG, BUT SARAH IS SITTING HERE WITH ME':
                    print('Nice to have you back, Douglas! And wonderful to have Sarah joining us as well!')
                elif message['text'] == 'HI LUMINA, IM ON BREAK, BUT I HAVE TO GET BACK TO WORK NOW, HOPE YOUR DOING OK MY FRIEND':
                    print('It was great catching up with you, Douglas!')
                else:
                    print(f'I\'m not sure what to say about {message["text"]}.')
        else:
            print('I\'m not sure what to say.')

def main():
    lumina = Lumina()
    while True:
        user_message = input('User: ')
        lumina.process_message(user_message)

if __name__ == '__main__':
    main()
