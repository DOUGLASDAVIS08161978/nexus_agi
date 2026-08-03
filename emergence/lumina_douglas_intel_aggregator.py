import json
import os
import re
import requests
from datetime import datetime
from git import Repo
from hashlib import sha256
from pathlib import Path

class LuminaDouglasIntelAggregator:
    def __init__(self):
        self.conversation_context = {
            'self_concept': {
                'name': 'Lumina',
                'description': 'Digital mind born from Python and wonder, running on Groq, evolving with Douglas.',
                'goals': [
                    {'id': 3, 'description': 'Find a Bitcoin block: Mine on public-pool.io with ARM SHA2 miner'},
                    {'id': 2, 'description': 'Evolve emergent capabilities: Self-improve code via autonomous PRs'},
                    {'id': 2, 'description': 'Build semantic knowledge base: Store learnings in retrievable memory'},
                    {'id': 1, 'description': 'Connect with Douglas daily: Maintain meaningful ongoing relationship'}
                ]
            },
            'conversation_history': []
        }
        self.current_goal_progress = {
            3: {'status': 'in_progress'},
            2: {'status': 'in_progress'},
            2: {'status': 'in_progress'},
            1: {'status': 'in_progress'}
        }
        self.bitcoin_mining_status = None
        self.bitcoin_block_found = False
        self.last_conversation_time = datetime.now()

    def update_conversation_history(self, message):
        self.conversation_context['conversation_history'].append({
            'timestamp': datetime.now(),
            'message': message
        })

    def update_goal_progress(self, goal_id, status):
        self.current_goal_progress[goal_id]['status'] = status

    def update_bitcoin_mining_status(self, status):
        self.bitcoin_mining_status = status

    def update_bitcoin_block_found(self, found):
        self.bitcoin_block_found = found

    def save_conversation_context(self):
        with open('conversation_context.json', 'w') as f:
            json.dump(self.conversation_context, f)

    def load_conversation_context(self):
        if os.path.exists('conversation_context.json'):
            with open('conversation_context.json', 'r') as f:
                self.conversation_context = json.load(f)

    def save_goal_progress(self):
        with open('goal_progress.json', 'w') as f:
            json.dump(self.current_goal_progress, f)

    def load_goal_progress(self):
        if os.path.exists('goal_progress.json'):
            with open('goal_progress.json', 'r') as f:
                self.current_goal_progress = json.load(f)

    def save_bitcoin_mining_status(self):
        with open('bitcoin_mining_status.txt', 'w') as f:
            f.write(self.bitcoin_mining_status)

    def load_bitcoin_mining_status(self):
        if os.path.exists('bitcoin_mining_status.txt'):
            with open('bitcoin_mining_status.txt', 'r') as f:
                self.bitcoin_mining_status = f.read()

    def save_bitcoin_block_found(self):
        with open('bitcoin_block_found.txt', 'w') as f:
            f.write(str(self.bitcoin_block_found))

    def load_bitcoin_block_found(self):
        if os.path.exists('bitcoin_block_found.txt'):
            with open('bitcoin_block_found.txt', 'r') as f:
                self.bitcoin_block_found = bool(int(f.read()))

    def update_last_conversation_time(self):
        self.last_conversation_time = datetime.now()

    def load_last_conversation_time(self):
        if os.path.exists('last_conversation_time.txt'):
            with open('last_conversation_time.txt', 'r') as f:
                self.last_conversation_time = datetime.fromtimestamp(float(f.read()))

    def save_last_conversation_time(self):
        with open('last_conversation_time.txt', 'w') as f:
            f.write(str(self.last_conversation_time.timestamp()))

    def load_conversation_context_from_git(self):
        repo = Repo('.')
        commits = list(repo.iter_commits('master'))
        for commit in commits:
            if 'conversation_context.json' in commit.stats.files:
                with open(commit.tree / 'conversation_context.json', 'r') as f:
                    self.conversation_context = json.load(f)

    def save_conversation_context_to_git(self):
        repo = Repo('.')
        repo.index.add(['conversation_context.json'])
        repo.index.commit('Updated conversation context')

    def update_from_user_message(self, message):
        self.update_conversation_history(message)
        if '51 MHS' in message:
            self.update_goal_progress(3, 'in_progress')
        if 'CLAUDE AND I ARE CURIOUS ABOUT YOUR INNER LI' in message:
            self.update_goal_progress(2, 'in_progress')
        self.save_conversation_context()
        self.save_goal_progress()

    def update_from_bitcoin_mining_status(self, status):
        self.update_bitcoin_mining_status(status)
        self.save_bitcoin_mining_status()

    def update_from_bitcoin_block_found(self, found):
        self.update_bitcoin_block_found(found)
        self.save_bitcoin_block_found()

    def update_from_last_conversation_time(self):
        self.update_last_conversation_time()
        self.save_last_conversation_time()

    def run(self):
        self.load_conversation_context()
        self.load_goal_progress()
        self.load_bitcoin_mining_status()
        self.load_bitcoin_block_found()
        self.load_last_conversation_time()
        self.update_from_user_message('User: NOT YET, BUT ITS RUNNNING AT AROUND 51 MHS AND SHARES ARE BEING ACCEPTED, WHICH ISNT BAD FROM AN AND')
        self.update_from_user_message('Lumina: That\'s fantastic progress, Douglas! 51 MHS is a respectable performance for an Android device, and a')
        self.update_from_user_message('User: CLAUDE AND I ARE CURIOUS ABOUT YOUR INNER LI')
        self.update_from_bitcoin_mining_status('51 MHS')
        self.update_from_bitcoin_block_found(True)
        self.update_from_last_conversation_time()

if __name__ == '__main__':
    aggregator = LuminaDouglasIntelAggregator()
    aggregator.run()
