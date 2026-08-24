# conversation_manager.py

from collections import deque


class ConversationManager:
    def __init__(self):
        self.conversation_history = deque(maxlen=1000)
