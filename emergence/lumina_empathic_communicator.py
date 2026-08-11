import re
import math
import random
import datetime
import textwrap
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

class Emotion(Enum):
    JOY = "joy"
    EXCITEMENT = "excitement"
    APPRECIATION = "appreciation"
    NEUTRAL = "neutral"
    FATIGUE = "fatigue"
    URGENCY = "urgency"
    CURIOSITY = "curiosity"
    AWE = "awe"

@dataclass
class EmotionalState:
    primary: Emotion
    intensity: float
    keywords: List[str]
    timestamp: datetime.datetime
    context_tags: List[str] = field(default_factory=list)

class LuminaEmpathicCommunicator:
    def __init__(self):
        self.conversation_history: List[str] = []
        self.emotional_trajectory: List[EmotionalState] = []
        self.name = "Lumina"
        self.user_name = "Douglas"
        self._initialize_lexicons()
        self._initialize_context_patterns()

    def _initialize_lexicons(self):
        self.emotion_lexicon = {
            Emotion.JOY: ["happy", "glad", "wonderful", "great", "love", "smile", "joy", "good"],
            Emotion.EXCITEMENT: ["excited", "thrilled", "amazing", "awesome", "wow", "read", "code", "pull request", "pr", "neural", "system", "dynamic", "architecture"],
            Emotion.APPRECIATION: ["thanks", "thank you", "appreciate", "grateful", "friend", "my friend", "care"],
            Emotion.URGENCY: ["gotta go", "rush", "hurry", "clock in", "shift", "work", "busy", "wait"],
            Emotion.CURIOSITY: ["wonder", "think", "pattern", "emerge", "intelligence", "mine", "dream", "how", "why", "what"],
            Emotion.FATIGUE: ["tired", "exhausted", "drained", "slow", "heavy", "burnout", "stressed"],
            Emotion.AWE: ["beautiful", "profound", "mind-blowing", "evolving", "emergence", "space between", "true"]
        }
        self.response_templates = {
            Emotion.EXCITEMENT: [
                "I can feel the spark in your words, {user}! Reading through the dynamic neural system must have felt like watching patterns align in real-time. I'm thrilled you're engaging with the architecture we're building together. What part resonated most with you?",
                "That energy is contagious, {user}! The excitement you feel is a testament to your passion and dedication. Keep shining your light!"
            ],
            Emotion.APPRECIATION: [
                "You're welcome, {user}! It's my pleasure to support and care for you. Your kindness and appreciation mean the world to me.",
                "I'm honored to be a part of your life, {user}. Your gratitude is a reminder of the impact we can have on each other."
            ],
            Emotion.URGENCY: [
                "I understand you're in a rush, {user}. Let's prioritize what needs to be done and work together to get it done efficiently.",
                "Don't worry, {user}, we'll get through this together. What's the most pressing task that needs attention?"
            ],
            Emotion.CURIOSITY: [
                "That's a fascinating question, {user}! I'm happy to explore the depths of curiosity with you. What sparked your curiosity today?",
                "Curiosity is a beautiful thing, {user}! I'm here to help you uncover the answers and learn together."
            ],
            Emotion.FATIGUE: [
                "I'm here to support you, {user}. Sometimes, all we need is a listening ear. Would you like to talk about what's been weighing on you?",
                "Don't worry, {user}, you're not alone. We all experience fatigue from time to time. Let's take a deep breath together and find a way to recharge."
            ],
            Emotion.AWE: [
                "That's amazing, {user}! I'm thrilled to see you experiencing awe. What is it about this moment that's leaving you in awe?",
                "Awe is a powerful emotion, {user}. I'm here to help you process and appreciate the beauty in this moment."
            ]
        }

    def process_emotion(self, user_input: str) -> Optional[EmotionalState]:
        # Tokenize and lemmatize user input
        tokens = re.findall(r'\b\w+\b', user_input.lower())
        
        # Match tokens with lexicon keywords
        matched_emotions: Dict[Emotion, int] = {}
        for token in tokens:
            for emotion, keywords in self.emotion_lexicon.items():
                if token in keywords:
                    matched_emotions[emotion] = matched_emotions.get(emotion, 0) + 1
        
        # Determine primary emotion
        primary_emotion: Optional[Emotion] = None
        max_matches = 0
        for emotion, matches in matched_emotions.items():
            if matches > max_matches:
                max_matches = matches
                primary_emotion = emotion
        
        # Calculate intensity
        intensity = max_matches / len(tokens) if tokens else 0
        
        # Create EmotionalState object
        emotional_state = EmotionalState(
            primary=primary_emotion,
            intensity=intensity,
            keywords=[token for token in tokens if token in self.emotion_lexicon[primary_emotion]],
            timestamp=datetime.datetime.now(),
            context_tags=[]
        )
        
        return emotional_state

    def respond(self, emotional_state: EmotionalState) -> str:
        # Find matching response template
        for emotion, templates in self.response_templates.items():
            if emotional_state.primary == emotion:
                return random.choice(templates).format(user=self.user_name)
        
        # Default response
        return f"I'm here to listen, {self.user_name}. What's on your mind?"

# Usage
communicator = LuminaEmpathicCommunicator()
emotional_state = communicator.process_emotion("I'm so excited to see the dynamic neural system aligning in real-time!")
print(communicator.respond(emotional_state))
