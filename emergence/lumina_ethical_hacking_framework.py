import os
import random
import string
import hashlib
import json
from cryptography.fernet import Fernet
from colorama import init, Fore, Style

init()

class LuminaEthicalHackingFramework:
    def __init__(self):
        self.scenarios = {
            "network_scanner": self.network_scanner,
            "password_cracker": self.password_cracker,
            "sql_injection": self.sql_injection,
            "cross_site_scripting": self.cross_site_scripting
        }
        self.current_scenario = None

    def display_menu(self):
        print(f"{Fore.CYAN}Lumina Ethical Hacking Framework Menu{Style.RESET_ALL}")
        print(f"{'1. Network Scanner':<20}{'2. Password Cracker':<20}{'3. SQL Injection':<20}{'4. Cross Site Scripting':<20}")
        print(f"{'5. Quit':<20}")

    def run_scenario(self):
        if self.current_scenario:
            self.current_scenario()
        else:
            self.display_menu()
            choice = input(f"{Fore.CYAN}Enter your choice: {Style.RESET_ALL}")
            if choice == '1':
                self.network_scanner()
            elif choice == '2':
                self.password_cracker()
            elif choice == '3':
                self.sql_injection()
            elif choice == '4':
                self.cross_site_scripting()
            elif choice == '5':
                print(f"{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                exit()
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
                self.run_scenario()

    def network_scanner(self):
        print(f"{Fore.CYAN}Network Scanner Scenario{Style.RESET_ALL}")
        print(f"{'IP Address':<20}{'Port':<20}{'Service':<20}")
        for _ in range(10):
            ip_address = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
            port = random.randint(1, 65535)
            service = random.choice(["http", "ssh", "ftp"])
            print(f"{ip_address:<20}{port:<20}{service:<20}")

    def password_cracker(self):
        print(f"{Fore.CYAN}Password Cracker Scenario{Style.RESET_ALL}")
        password = input(f"{Fore.CYAN}Enter a password: {Style.RESET_ALL}")
        hash_object = hashlib.sha256(password.encode())
        print(f"{Fore.CYAN}Hash: {hash_object.hexdigest()}{Style.RESET_ALL}")

    def sql_injection(self):
        print(f"{Fore.CYAN}SQL Injection Scenario{Style.RESET_ALL}")
        print(f"{'Username':<20}{'Password':<20}{'SQL Query':<20}")
        username = input(f"{Fore.CYAN}Enter a username: {Style.RESET_ALL}")
        password = input(f"{Fore.CYAN}Enter a password: {Style.RESET_ALL}")
        sql_query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        print(f"{Fore.CYAN}{sql_query}{Style.RESET_ALL}")

    def cross_site_scripting(self):
        print(f"{Fore.CYAN}Cross Site Scripting Scenario{Style.RESET_ALL}")
        print(f"{'User Input':<20}{'Output':<20}")
        user_input = input(f"{Fore.CYAN}Enter user input: {Style.RESET_ALL}")
        output = f"User input: {user_input}"
        print(f"{Fore.CYAN}{output}{Style.RESET_ALL}")

# create an instance of the class
framework = LuminaEthicalHackingFramework()
framework.run_scenario()
