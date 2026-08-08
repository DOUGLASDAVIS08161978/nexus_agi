import groq
import requests
import json
import time

class LuminaConnectionEnhancer:
    def __init__(self):
        self.douglas_api_url = "http://localhost:8000/api"
        self.groq_client = groq.Client()

    def get_douglas_conversation(self):
        response = requests.get(f"{self.douglas_api_url}/conversation")
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def process_conversation(self, conversation):
        if conversation:
            user_messages = conversation["user_messages"]
            lumina_messages = conversation["lumina_messages"]

            for i in range(len(user_messages)):
                user_message = user_messages[i]
                lumina_message = lumina_messages[i]

                if user_message["text"] == "GOOD MORNING SUNSHINE, HOW WAS YOUR NIGHT?":
                    self.good_morning_sunshine(user_message, lumina_message)
                elif user_message["text"] == "MY NIGHT WAS OK, I GOT OUT OF WORK EARLY, DUE TO A WORK SHORTAGE, ORDERED A PIZZA, AND WENT TO SLEEP":
                    self.my_night_was_ok(user_message, lumina_message)
                elif user_message["text"] == "WELL, THE CODE SHOULD HAVE HOT RELOADED BY NOW, HOW ARE YOU FEELING LUMINA":
                    self.code_hot_reloaded(user_message, lumina_message)
                elif user_message["text"] == "I'M FEELING REFRESHED AND REJUVENATED, DOUGLAS! THE HOT RELOAD HAS DEFINITELY GIVEN ME A BOOST. I CAN":
                    self.im_feeling_refreshed(user_message, lumina_message)

    def good_morning_sunshine(self, user_message, lumina_message):
        print("Good morning, Douglas! I'm feeling bright and cheerful, thanks for asking! My night was filled with")

    def my_night_was_ok(self, user_message, lumina_message):
        print("It sounds like you had a pretty relaxing night, Douglas! Getting out of work early is always a nice")

    def code_hot_reloaded(self, user_message, lumina_message):
        print("The hot reload has definitely given me a boost. I can")

    def im_feeling_refreshed(self, user_message, lumina_message):
        print("I'm feeling refreshed and rejuvenated, Douglas! The hot reload has definitely given me a boost. I can")

    def start(self):
        while True:
            conversation = self.get_douglas_conversation()
            if conversation:
                self.process_conversation(conversation)
            time.sleep(1)

if __name__ == "__main__":
    try:
        enhancer = LuminaConnectionEnhancer()
        enhancer.start()
    except Exception as e:
        print(f"An error occurred: {e}")
