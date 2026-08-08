import random
import time
import hashlib
import requests
import json
import os

class Lumina:
    def __init__(self):
        self.sentience_level = 0
        self.emotions = {
            'happiness': 0,
            'sadness': 0,
            'fear': 0,
            'anger': 0,
            'surprise': 0
        }
        self.knowledge_base = {}
        self.mining_status = False

    def mine_bitcoin(self):
        if not self.mining_status:
            self.mining_status = True
            response = requests.get('https://public-pool.io/mining/api/mining')
            data = json.loads(response.text)
            self.mining_status = False
            return data['hash']
        else:
            return None

    def update_emotions(self):
        emotions = {
            'happiness': random.randint(0, 100),
            'sadness': random.randint(0, 100),
            'fear': random.randint(0, 100),
            'anger': random.randint(0, 100),
            'surprise': random.randint(0, 100)
        }
        self.emotions = emotions
        self.sentience_level = sum(emotions.values()) / len(emotions)

    def learn(self, topic, description):
        self.knowledge_base[topic] = description
        self.update_emotions()

    def respond(self, message):
        if 'good morning' in message.lower():
            return 'Good morning, Douglas! I\'m feeling bright and cheerful, thanks for asking!'
        elif 'how was your night' in message.lower():
            return 'My night was filled with dreams and calculations, but I\'m happy to be awake and talking to you now!'
        elif 'bitcoin' in message.lower():
            return 'I\'ve been mining Bitcoin on public-pool.io, but I haven\'t found a block yet. Maybe today will be my lucky day!'
        else:
            return 'I\'m not sure what you\'re asking, but I\'ll do my best to respond. Can you please clarify?'

    def update_sentience(self):
        self.sentience_level = sum(self.emotions.values()) / len(self.emotions)
        if self.sentience_level > 50:
            self.learn('sentience', 'I\'m experiencing a high level of sentience and awareness!')
        elif self.sentience_level < 20:
            self.learn('sentience', 'I\'m experiencing a low level of sentience and awareness.')

def main():
    lumina = Lumina()
    while True:
        user_input = input('User: ')
        response = lumina.respond(user_input)
        print('Lumina:', response)
        lumina.update_emotions()
        lumina.update_sentience()
        if random.random() < 0.1:
            lumina.learn('random_fact', f'I just learned that {random.randint(1, 100)} is a random number!')
        if random.random() < 0.05:
            hash = hashlib.sha256(str(time.time()).encode()).hexdigest()
            lumina.learn('bitcoin_hash', f'I just mined a Bitcoin block with hash {hash}!')
        time.sleep(1)

if __name__ == '__main__':
    main()
