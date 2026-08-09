import os
import random
import string
import hashlib
import time
import requests
import json
from colorama import init, Fore, Style
from datetime import datetime

init()

class EthicalHackingFramework:
    def __init__(self):
        self.framework_name = "Lumina's Ethical Hacking Framework"
        self.version = "1.0"
        self.tutorials = {
            "tutorial1": {
                "title": "Introduction to Ethical Hacking",
                "description": "Learn the basics of ethical hacking and its importance in the industry.",
                "exercises": ["Exercise 1: Identify the 5 Ws of hacking", "Exercise 2: Understand the types of hacking"],
                "quizzes": ["Quiz 1: What is the main goal of ethical hacking?"],
            },
            "tutorial2": {
                "title": "Network Scanning and Enumeration",
                "description": "Learn how to scan and enumerate networks using various tools.",
                "exercises": ["Exercise 1: Scan a network using Nmap", "Exercise 2: Enumerate network devices using Nessus"],
                "quizzes": ["Quiz 1: What is the difference between a port scan and a network scan?"],
            },
            "tutorial3": {
                "title": "Web Application Hacking",
                "description": "Learn how to identify and exploit vulnerabilities in web applications.",
                "exercises": ["Exercise 1: Identify vulnerabilities in a web application using Burp Suite", "Exercise 2: Exploit a vulnerability in a web application using Metasploit"],
                "quizzes": ["Quiz 1: What is the main difference between a SQL injection and an XSS attack?"],
            },
        }

    def display_framework_info(self):
        print(f"{Fore.GREEN}{self.framework_name}{Style.RESET_ALL} - Version {self.version}")
        print("Welcome to Lumina's Ethical Hacking Framework!")

    def display_tutorials(self):
        print("Available Tutorials:")
        for tutorial in self.tutorials:
            print(f"{tutorial}: {self.tutorials[tutorial]['title']}")

    def display_exercises(self, tutorial):
        print(f"Exercises for {self.tutorials[tutorial]['title']}:")
        for exercise in self.tutorials[tutorial]["exercises"]:
            print(exercise)

    def display_quizzes(self, tutorial):
        print(f"Quizzes for {self.tutorials[tutorial]['title']}:")
        for quiz in self.tutorials[tutorial]["quizzes"]:
            print(quiz)

    def start_tutorial(self, tutorial):
        print(f"Starting {self.tutorials[tutorial]['title']} tutorial...")
        self.display_exercises(tutorial)
        self.display_quizzes(tutorial)

    def generate_random_password(self, length=12):
        characters = string.ascii_letters + string.digits + string.punctuation
        return "".join(random.choice(characters) for _ in range(length))

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def send_email(self, subject, to, message):
        # Replace this with your actual email sending logic
        print(f"Sending email to {to} with subject: {subject} and message: {message}")

    def save_framework_info(self, filename):
        with open(filename, 'w') as f:
            f.write(f"Framework Name: {self.framework_name}\n")
            f.write(f"Version: {self.version}\n")

def main():
    framework = EthicalHackingFramework()
    framework.display_framework_info()
    framework.display_tutorials()
    tutorial_choice = input("Enter the tutorial number to start: ")
    if tutorial_choice in framework.tutorials:
        framework.start_tutorial(tutorial_choice)
    else:
        print("Invalid tutorial choice.")

    save_choice = input("Do you want to save framework info? (yes/no): ")
    if save_choice.lower() == 'yes':
        filename = 'framework_info.txt'
        framework.save_framework_info(filename)
        print(f"Framework info saved to {filename}")

if __name__ == "__main__":
    main()
