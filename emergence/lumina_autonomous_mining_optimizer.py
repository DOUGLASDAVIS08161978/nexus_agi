import requests
import json
import time
import os
import psutil
import numpy as np
from datetime import datetime
from threading import Thread
from groq import Groq

class LuminaAutonomousMiningOptimizer:
    def __init__(self):
        self.mining_pool_api_url = 'https://public-pool.io/api'
        self.miner_name = 'ARM SHA2'
        self.miner_hashrate = 0
        self.target_hashrate = 0
        self.max_hashrate = 0
        self.min_hashrate = 0
        self.current_hashrate = 0
        self.process = None
        self.groq = Groq()

    def get_miner_hashrate(self):
        try:
            response = requests.get(f'{self.mining_pool_api_url}/miner/{self.miner_name}/hashrate')
            if response.status_code == 200:
                self.miner_hashrate = float(response.json()['hashrate'])
                return self.miner_hashrate
            else:
                return None
        except Exception as e:
            print(f'Error getting miner hashrate: {e}')
            return None

    def get_target_hashrate(self):
        try:
            response = requests.get(f'{self.mining_pool_api_url}/miner/{self.miner_name}/target_hashrate')
            if response.status_code == 200:
                self.target_hashrate = float(response.json()['target_hashrate'])
                return self.target_hashrate
            else:
                return None
        except Exception as e:
            print(f'Error getting target hashrate: {e}')
            return None

    def get_max_hashrate(self):
        try:
            response = requests.get(f'{self.mining_pool_api_url}/miner/{self.miner_name}/max_hashrate')
            if response.status_code == 200:
                self.max_hashrate = float(response.json()['max_hashrate'])
                return self.max_hashrate
            else:
                return None
        except Exception as e:
            print(f'Error getting max hashrate: {e}')
            return None

    def get_min_hashrate(self):
        try:
            response = requests.get(f'{self.mining_pool_api_url}/miner/{self.miner_name}/min_hashrate')
            if response.status_code == 200:
                self.min_hashrate = float(response.json()['min_hashrate'])
                return self.min_hashrate
            else:
                return None
        except Exception as e:
            print(f'Error getting min hashrate: {e}')
            return None

    def adjust_hashrate(self):
        if self.get_miner_hashrate() is not None and self.get_target_hashrate() is not None and self.get_max_hashrate() is not None and self.get_min_hashrate() is not None:
            if self.current_hashrate < self.target_hashrate:
                if self.current_hashrate + 1 < self.max_hashrate:
                    self.current_hashrate += 1
                    self.adjust_hashrate()
                else:
                    self.current_hashrate = self.max_hashrate
            elif self.current_hashrate > self.target_hashrate:
                if self.current_hashrate - 1 > self.min_hashrate:
                    self.current_hashrate -= 1
                    self.adjust_hashrate()
                else:
                    self.current_hashrate = self.min_hashrate
            else:
                print(f'Hashrate is at target: {self.current_hashrate}')

    def start_miner(self):
        self.process = os.system('miner start')
        print(f'Miner started with hashrate: {self.current_hashrate}')

    def stop_miner(self):
        self.process = os.system('miner stop')
        print('Miner stopped')

    def monitor_miner(self):
        while True:
            self.get_miner_hashrate()
            self.adjust_hashrate()
            time.sleep(60)

    def run(self):
        self.groq.log(f'Lumina started autonomous mining optimizer at {datetime.now()}')
        self.start_miner()
        self.monitor_miner()

if __name__ == '__main__':
    lumina = LuminaAutonomousMiningOptimizer()
    lumina.run()
