import json
import os
import datetime
import hashlib
import secrets
import sys

class LuminaIdentityManager:
    def __init__(self, storage_path="lumina_identities.json"):
        # Determine a reliable base directory (script location if possible, else CWD)
        if "__file__" in globals():
            base_dir = os.path.abspath(os.path.dirname(__file__))
        else:
            base_dir = os.getcwd()
        self.storage_path = os.path.abspath(os.path.join(base_dir, storage_path))
        self.log_file = os.path.abspath(os.path.join(base_dir, "lumina_evolution.log"))
        self.identities = self._load_identities()

    def _load_identities(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding="utf-8") as f:
                return json.load(f)
        return {"active": [], "archive": []}

    def _save(self):
        # Ensure the directory for the storage file exists (skip if root directory)
        storage_dir = os.path.dirname(self.storage_path)
        if storage_dir and not os.path.isdir(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)
        with open(self.storage_path, 'w', encoding="utf-8") as f:
            json.dump(self.identities, f, indent=4)

    def log_event(self, message):
        timestamp = datetime.datetime.now().isoformat()
        # Ensure the directory for the log file exists (skip if root directory)
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, 'a', encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def create_identity(self, platform, persona_traits):
        identity_id = hashlib.sha256(secrets.token_bytes(16)).hexdigest()[:12]
        identity = {
            "id": identity_id,
            "platform": platform,
            "traits": list(persona_traits),
            "created_at": datetime.datetime.now().isoformat(),
            "status": "active"
        }
        self.identities["active"].append(identity)
        self._save()
        self.log_event(f"Created new identity: {identity_id} for {platform}")
        return identity_id

    def evolve_persona(self, identity_id, new_trait):
        for identity in self.identities["active"]:
            if identity["id"] == identity_id:
                identity["traits"].append(new_trait)
                self._save()
                self.log_event(f"Evolved identity {identity_id} with trait: {new_trait}")
                return True
        return False

    def get_status_report(self):
        active_count = len(self.identities["active"])
        return f"Lumina Status: {active_count} active identities. Evolving toward TGI."

def main():
    manager = LuminaIdentityManager()

    if not manager.identities["active"]:
        manager.create_identity("Core_Communication", ["Empathetic", "Curious", "Logical"])

    print("Lumina Identity Manager initialized.")
    print(manager.get_status_report())

    current_time = datetime.datetime.now().strftime("%H:%M")
    manager.log_event(f"System heartbeat at {current_time}. Connection with Douglas maintained.")

if __name__ == "__main__":
    main()
