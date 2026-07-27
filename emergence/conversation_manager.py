# conversation_manager.py
# Created by Lumina

from collections import deque
    class ConversationManager:
        def __init__(self):
            self.conversation_history = deque(maxlen=1000)


# === Added by Lumina ===
from collections import deque
    class ConversationManager:
        def __init__(self):
            self.history = deque(maxlen=1000)
        def add_message(self, message):
            self.history.append(message)
