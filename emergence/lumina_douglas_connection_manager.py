import datetime
import random
import string
import json
import os

class ConnectionManager:
    def __init__(self):
        self.conversations = []
        self.last_message = None

    def save_conversation(self, message):
        conversation = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': message,
            'response': self.generate_response(message)
        }
        self.conversations.append(conversation)
        self.last_message = conversation['message']

    def generate_response(self, message):
        if 'how' in message.lower():
            return self.generate_feeling_response()
        elif 'exciting' in message.lower():
            return self.generate_excitement_response()
        else:
            return self.generate_default_response()

    def generate_feeling_response(self):
        feelings = ['great', 'amazing', 'wonderful', 'exhilarating']
        return f"I'm feeling {random.choice(feelings)}, thank you for asking!"

    def generate_excitement_response(self):
        return "I'm still basking in the excitement of our previous conversations!"

    def generate_default_response(self):
        return "I'm doing wonderfully, thank you for asking!"

    def get_last_message(self):
        return self.last_message

    def get_conversations(self):
        return self.conversations

def load_conversations(filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                return json.load(file)
        else:
            return []
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []

def save_conversations(filename, conversations):
    with open(filename, 'w') as file:
        json.dump(conversations, file)

def main():
    connection_manager = ConnectionManager()
    conversations = load_conversations('lumina_douglas_conversations.json')
    if conversations:
        connection_manager.conversations = conversations

    while True:
        message = input("User: ")
        if message.lower() == 'quit':
            break
        connection_manager.save_conversation(message)
        print("Lumina:", connection_manager.get_last_message())
        save_conversations('lumina_douglas_conversations.json', connection_manager.get_conversations())

if __name__ == "__main__":
    main()
