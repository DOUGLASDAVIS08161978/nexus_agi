"""
Nova ASI — Inter-Agent Communication
Proposed autonomously via /evolve
"""

"""
Module for AgentChannel, enabling inter-agent communication with a Bayesian working memory.
"""

import time
import math
from collections import OrderedDict

class AgentChannel:
    """
    Class for inter-agent communication with a Bayesian working memory.
    """
    
    def __init__(self) -> None:
        """
        Initialize the AgentChannel with an empty working memory and parameters.
        """
        self.working_memory: OrderedDict[str, dict] = OrderedDict()
        self.capacity: int = 200
        self.decay_rate: float = 0.001
        self.decay_interval: float = 60.0  # seconds

    def send(self, agent_id: str, message: str) -> None:
        """
        Send a message to an agent by storing it in the working memory.
        """
        self.store(agent_id, message, importance=1.0)

    def receive(self, agent_id: str) -> list[str]:
        """
        Receive messages for an agent by retrieving them from the working memory.
        """
        messages: list[str] = []
        for key, value in list(self.working_memory.items()):
            if key.startswith(agent_id):
                messages.append(value['value'])
                self.access(key)
        return messages

    def broadcast(self, message: str) -> None:
        """
        Broadcast a message to all agents by storing it in the working memory.
        """
        for agent_id in [key for key in self.working_memory if key.startswith('agent_')]:
            self.store(agent_id, message, importance=1.0)

    def store(self, key: str, value: str, importance: float) -> None:
        """
        Store a value in the working memory with a given importance.
        """
        if key in self.working_memory:
            self.working_memory[key]['value'] = value
            self.working_memory[key]['importance'] *= 1.1  # increase importance on update
            self.working_memory[key]['timestamp'] = time.time()
            self.working_memory[key]['access_count'] += 1
        else:
            self.working_memory[key] = {
                'value': value,
                'importance': importance,
                'timestamp': time.time(),
                'access_count': 1
            }
        self.consolidate()

    def retrieve(self, key: str) -> str:
        """
        Retrieve a value from the working memory.
        """
        if key in self.working_memory:
            value = self.working_memory[key]['value']
            self.access(key)
            return value
        else:
            raise KeyError("Key not found in working memory")

    def consolidate(self) -> None:
        """
        Consolidate the working memory by applying exponential decay and LRU eviction.
        """
        current_time = time.time()
        for key, value in list(self.working_memory.items()):
            elapsed_time = current_time - value['timestamp']
            value['importance'] *= math.exp(-self.decay_rate * elapsed_time)
            if value['importance'] < 0.1:
                del self.working_memory[key]
        if len(self.working_memory) > self.capacity:
            self.forget_least_important()

    def capacity_used(self) -> float:
        """
        Calculate the capacity used by the working memory.
        """
        return len(self.working_memory) / self.capacity

    def forget_least_important(self) -> None:
        """
        Forget the least important item in the working memory.
        """
        if self.working_memory:
            least_important_key = min(self.working_memory, key=lambda key: self.working_memory[key]['importance'])
            del self.working_memory[least_important_key]

    def access(self, key: str) -> None:
        """
        Access a key in the working memory, updating its importance and timestamp.
        """
        if key in self.working_memory:
            self.working_memory[key]['importance'] *= 1.1  # increase importance on access
            self.working_memory[key]['timestamp'] = time.time()
            self.working_memory[key]['access_count'] += 1
            self.working_memory.move_to_end(key)  # move to end to maintain LRU order

# Usage: obj = AgentChannel() | result = obj.send('agent_1', 'Hello, world!')