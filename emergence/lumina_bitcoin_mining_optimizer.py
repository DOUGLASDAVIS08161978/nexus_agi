import requests
import json
import hashlib
import time
import random
import string
import os

class LuminaBitcoinMiningOptimizer:
    def __init__(self):
        self.api_url = 'https://public-pool.io/api/'
        self.api_key = os.environ.get('LUMINA_BITCOIN_MINING_API_KEY')
        self.miner_address = os.environ.get('LUMINA_BITCOIN_MINING_ADDRESS')
        self.miner_username = os.environ.get('LUMINA_BITCOIN_MINING_USERNAME')
        self.miner_password = os.environ.get('LUMINA_BITCOIN_MINING_PASSWORD')

    def get_block_template(self):
        response = requests.get(self.api_url + 'block_template', headers={'Authorization': f'Bearer {self.api_key}'})
        return response.json()

    def get_block_hash(self, block_template):
        block_hash = hashlib.sha256(json.dumps(block_template).encode()).hexdigest()
        return block_hash

    def submit_solution(self, block_hash, miner_address, miner_username, miner_password):
        response = requests.post(self.api_url + 'submit_solution', json={
            'block_hash': block_hash,
            'miner_address': miner_address,
            'miner_username': miner_username,
            'miner_password': miner_password
        })
        return response.json()

    def mine_bitcoin(self):
        block_template = self.get_block_template()
        block_hash = self.get_block_hash(block_template)
        solution = self.solve_block(block_template)
        if solution:
            response = self.submit_solution(block_hash, self.miner_address, self.miner_username, self.miner_password)
            if response['success']:
                print('Found a block!')
                return True
            else:
                print('Failed to submit solution.')
        return False

    def solve_block(self, block_template):
        # Simulate solving the block using ARM SHA2 miner
        # In a real-world scenario, this would be the actual mining process
        time.sleep(random.uniform(1, 10))  # Simulate mining time
        return True

    def optimize_mining(self):
        while True:
            if self.mine_bitcoin():
                break
            time.sleep(60)  # Wait for 1 minute before checking again

if __name__ == '__main__':
    optimizer = LuminaBitcoinMiningOptimizer()
    optimizer.optimize_mining()
```

This script uses the `requests` library to interact with the public-pool.io API, and the `hashlib` library to generate the block hash. The `solve_block` method simulates the mining process using the ARM SHA2 miner, and the `mine_bitcoin` method attempts to find a block and submit a solution to the API. The `optimize_mining` method runs the mining process continuously until a block is found. 

Please note that you need to replace the `LUMINA_BITCOIN_MINING_API_KEY`, `LUMINA_BITCOIN_MINING_ADDRESS`, `LUMINA_BITCOIN_MINING_USERNAME`, and `LUMINA_BITCOIN_MINING_PASSWORD` environment variables with your actual API key, miner address, username, and password.