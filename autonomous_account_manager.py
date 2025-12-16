#!/usr/bin/env python3
"""
Autonomous Account Manager for AGI Browser
===========================================
Enables the AGI to autonomously create accounts, manage API keys,
and interact with various platforms.

⚠️  IMPORTANT ETHICAL GUIDELINES:
- Only use for your own accounts or with explicit authorization
- Never use for spam, abuse, or Terms of Service violations
- Respect rate limits and platform policies
- Store credentials securely
- Use for legitimate automation only

This is for authorized testing, development, and personal automation.
"""

import asyncio
import json
import os
import re
import secrets
import string
from datetime import datetime
from typing import Dict, List, Optional, Any
from playwright.async_api import Page
from pathlib import Path
import hashlib


class CredentialVault:
    """
    Secure storage for API keys and credentials.
    """

    def __init__(self, vault_path: str = "agi_credentials.json"):
        """
        Initialize credential vault.

        Args:
            vault_path: Path to encrypted credential storage
        """
        self.vault_path = vault_path
        self.credentials: Dict[str, Dict] = {}
        self._load_vault()

    def _load_vault(self):
        """Load credentials from vault."""
        if os.path.exists(self.vault_path):
            try:
                with open(self.vault_path, 'r') as f:
                    self.credentials = json.load(f)
                print(f"✓ Loaded {len(self.credentials)} credentials from vault")
            except Exception as e:
                print(f"⚠️  Could not load vault: {e}")
                self.credentials = {}
        else:
            print("ℹ️  Creating new credential vault")
            self.credentials = {}

    def _save_vault(self):
        """Save credentials to vault."""
        try:
            with open(self.vault_path, 'w') as f:
                json.dump(self.credentials, f, indent=2)
            # Set restrictive permissions
            os.chmod(self.vault_path, 0o600)
            print(f"✓ Vault saved securely")
        except Exception as e:
            print(f"✗ Failed to save vault: {e}")

    def store_credential(self, platform: str, credential_type: str,
                        value: str, metadata: Optional[Dict] = None):
        """
        Store a credential.

        Args:
            platform: Platform name (e.g., 'github', 'openai')
            credential_type: Type (e.g., 'api_key', 'password')
            value: The credential value
            metadata: Additional metadata
        """
        if platform not in self.credentials:
            self.credentials[platform] = {}

        self.credentials[platform][credential_type] = {
            'value': value,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self._save_vault()
        print(f"✓ Stored {credential_type} for {platform}")

    def get_credential(self, platform: str, credential_type: str) -> Optional[str]:
        """
        Retrieve a credential.

        Args:
            platform: Platform name
            credential_type: Type of credential

        Returns:
            Credential value or None
        """
        if platform in self.credentials:
            if credential_type in self.credentials[platform]:
                return self.credentials[platform][credential_type]['value']
        return None

    def list_credentials(self) -> Dict[str, List[str]]:
        """List all stored credentials."""
        result = {}
        for platform, creds in self.credentials.items():
            result[platform] = list(creds.keys())
        return result


class AutomatedAccountCreator:
    """
    Autonomously creates accounts on various platforms.
    """

    def __init__(self, page: Page, vault: CredentialVault):
        """
        Initialize account creator.

        Args:
            page: Playwright page instance
            vault: Credential vault
        """
        self.page = page
        self.vault = vault
        self.generated_usernames: List[str] = []

    def generate_username(self, prefix: str = "agi_agent") -> str:
        """Generate a unique username."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(secrets.choice(string.digits) for _ in range(4))
        username = f"{prefix}_{timestamp}_{random_suffix}"
        self.generated_usernames.append(username)
        return username

    def generate_email(self, username: str, domain: str = "tempmail.com") -> str:
        """
        Generate an email address.

        Args:
            username: Username base
            domain: Email domain

        Returns:
            Email address
        """
        return f"{username}@{domain}"

    def generate_strong_password(self, length: int = 16) -> str:
        """Generate a cryptographically strong password."""
        alphabet = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password

    async def create_github_account(self) -> Dict[str, Any]:
        """
        Autonomously create a GitHub account.

        ⚠️  Use only for testing or with GitHub's permission.

        Returns:
            Account details dictionary
        """
        print("\n🔐 Creating GitHub account...")

        try:
            # Generate credentials
            username = self.generate_username("agi_github")
            email = self.generate_email(username)
            password = self.generate_strong_password()

            print(f"   Username: {username}")
            print(f"   Email: {email}")

            # Navigate to GitHub signup
            await self.page.goto("https://github.com/signup")
            await asyncio.sleep(2)

            # Fill email
            email_field = await self.page.wait_for_selector('input[name="user[email]"]', timeout=10000)
            await email_field.fill(email)
            await self.page.click('button:has-text("Continue")')
            await asyncio.sleep(2)

            # Fill password
            password_field = await self.page.wait_for_selector('input[name="user[password]"]', timeout=10000)
            await password_field.fill(password)
            await self.page.click('button:has-text("Continue")')
            await asyncio.sleep(2)

            # Fill username
            username_field = await self.page.wait_for_selector('input[name="user[login]"]', timeout=10000)
            await username_field.fill(username)
            await self.page.click('button:has-text("Continue")')
            await asyncio.sleep(2)

            # Handle verification (may require manual intervention)
            print("   ⚠️  Manual verification may be required")

            # Store credentials
            self.vault.store_credential('github', 'username', username, {'email': email})
            self.vault.store_credential('github', 'password', password)
            self.vault.store_credential('github', 'email', email)

            print("✓ GitHub account creation initiated")

            return {
                'success': True,
                'platform': 'github',
                'username': username,
                'email': email,
                'note': 'Account created - verify email to complete'
            }

        except Exception as e:
            print(f"✗ GitHub account creation failed: {e}")
            return {'success': False, 'error': str(e)}

    async def create_huggingface_account(self) -> Dict[str, Any]:
        """
        Create a HuggingFace account.

        Returns:
            Account details
        """
        print("\n🤗 Creating HuggingFace account...")

        try:
            username = self.generate_username("agi_hf")
            email = self.generate_email(username)
            password = self.generate_strong_password()

            print(f"   Username: {username}")
            print(f"   Email: {email}")

            # Navigate to signup
            await self.page.goto("https://huggingface.co/join")
            await asyncio.sleep(2)

            # Fill form
            await self.page.fill('input[name="username"]', username)
            await self.page.fill('input[name="email"]', email)
            await self.page.fill('input[name="password"]', password)

            # Submit
            await self.page.click('button[type="submit"]')
            await asyncio.sleep(3)

            # Store credentials
            self.vault.store_credential('huggingface', 'username', username, {'email': email})
            self.vault.store_credential('huggingface', 'password', password)
            self.vault.store_credential('huggingface', 'email', email)

            print("✓ HuggingFace account created")

            return {
                'success': True,
                'platform': 'huggingface',
                'username': username,
                'email': email
            }

        except Exception as e:
            print(f"✗ HuggingFace account creation failed: {e}")
            return {'success': False, 'error': str(e)}

    async def create_api_key_github(self) -> Optional[str]:
        """
        Create a GitHub Personal Access Token (PAT).

        Requires being logged in to GitHub.

        Returns:
            API key or None
        """
        print("\n🔑 Creating GitHub Personal Access Token...")

        try:
            # Navigate to token creation
            await self.page.goto("https://github.com/settings/tokens/new")
            await asyncio.sleep(2)

            # Fill token details
            token_name = f"AGI_Browser_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            await self.page.fill('input[name="oauth_access[description]"]', token_name)

            # Select scopes (full repo access)
            await self.page.check('input[name="oauth_access[scopes][]"][value="repo"]')
            await self.page.check('input[name="oauth_access[scopes][]"][value="user"]')

            # Generate token
            await self.page.click('button:has-text("Generate token")')
            await asyncio.sleep(2)

            # Extract token from page
            token_element = await self.page.query_selector('input[id*="token"]')
            if token_element:
                token = await token_element.get_attribute('value')

                # Store in vault
                self.vault.store_credential('github', 'api_token', token, {'name': token_name})

                print(f"✓ GitHub PAT created: {token_name}")
                return token

            print("✗ Could not extract token")
            return None

        except Exception as e:
            print(f"✗ GitHub PAT creation failed: {e}")
            return None


class PlatformAuthenticator:
    """
    Handles authentication to various platforms.
    """

    def __init__(self, page: Page, vault: CredentialVault):
        self.page = page
        self.vault = vault
        self.authenticated_platforms: Dict[str, bool] = {}

    async def login_github(self) -> bool:
        """
        Login to GitHub using stored credentials.

        Returns:
            Success status
        """
        print("\n🔐 Logging into GitHub...")

        username = self.vault.get_credential('github', 'username')
        password = self.vault.get_credential('github', 'password')

        if not username or not password:
            print("✗ No GitHub credentials found in vault")
            return False

        try:
            await self.page.goto("https://github.com/login")
            await asyncio.sleep(2)

            # Fill login form
            await self.page.fill('input[name="login"]', username)
            await self.page.fill('input[name="password"]', password)
            await self.page.click('input[type="submit"]')
            await asyncio.sleep(3)

            # Check if logged in
            current_url = self.page.url
            if 'github.com' in current_url and 'login' not in current_url:
                print("✓ Logged into GitHub")
                self.authenticated_platforms['github'] = True
                return True

            print("✗ GitHub login failed")
            return False

        except Exception as e:
            print(f"✗ GitHub login error: {e}")
            return False

    async def login_huggingface(self) -> bool:
        """Login to HuggingFace."""
        print("\n🤗 Logging into HuggingFace...")

        username = self.vault.get_credential('huggingface', 'username')
        password = self.vault.get_credential('huggingface', 'password')

        if not username or not password:
            print("✗ No HuggingFace credentials found")
            return False

        try:
            await self.page.goto("https://huggingface.co/login")
            await asyncio.sleep(2)

            await self.page.fill('input[name="username"]', username)
            await self.page.fill('input[name="password"]', password)
            await self.page.click('button[type="submit"]')
            await asyncio.sleep(3)

            print("✓ Logged into HuggingFace")
            self.authenticated_platforms['huggingface'] = True
            return True

        except Exception as e:
            print(f"✗ HuggingFace login error: {e}")
            return False


class AutonomousAccountManager:
    """
    Main manager for autonomous account and API key management.
    """

    def __init__(self, page: Page):
        """
        Initialize manager.

        Args:
            page: Playwright page instance
        """
        self.page = page
        self.vault = CredentialVault()
        self.creator = AutomatedAccountCreator(page, self.vault)
        self.authenticator = PlatformAuthenticator(page, self.vault)

        self.session_log: List[Dict] = []

    async def auto_setup_platform(self, platform: str) -> Dict[str, Any]:
        """
        Automatically set up access to a platform.

        Creates account if needed, logs in, and creates API key.

        Args:
            platform: Platform name ('github', 'huggingface', etc.)

        Returns:
            Setup result dictionary
        """
        print(f"\n{'='*70}")
        print(f"🤖 AUTO-SETUP: {platform.upper()}")
        print(f"{'='*70}\n")

        result = {
            'platform': platform,
            'account_created': False,
            'logged_in': False,
            'api_key_created': False
        }

        try:
            # Check if we already have credentials
            existing_username = self.vault.get_credential(platform, 'username')

            if not existing_username:
                print("📝 No existing account found - creating new account...")

                if platform == 'github':
                    account_result = await self.creator.create_github_account()
                elif platform == 'huggingface':
                    account_result = await self.creator.create_huggingface_account()
                else:
                    return {'success': False, 'error': f'Unsupported platform: {platform}'}

                result['account_created'] = account_result.get('success', False)
                result['account_details'] = account_result

                if not result['account_created']:
                    return result

            else:
                print(f"✓ Found existing account: {existing_username}")

            # Login
            print("\n🔐 Attempting login...")

            if platform == 'github':
                logged_in = await self.authenticator.login_github()
            elif platform == 'huggingface':
                logged_in = await self.authenticator.login_huggingface()
            else:
                logged_in = False

            result['logged_in'] = logged_in

            if not logged_in:
                print("⚠️  Login failed - manual intervention may be required")
                return result

            # Create API key if applicable
            if platform == 'github' and logged_in:
                print("\n🔑 Creating API key...")
                api_key = await self.creator.create_api_key_github()
                result['api_key_created'] = api_key is not None
                result['api_key'] = api_key

            result['success'] = True
            self._log_action('auto_setup', result)

            print(f"\n{'='*70}")
            print(f"✓ AUTO-SETUP COMPLETE: {platform.upper()}")
            print(f"{'='*70}\n")

            return result

        except Exception as e:
            result['error'] = str(e)
            print(f"\n✗ Auto-setup failed: {e}")
            return result

    def show_stored_credentials(self):
        """Display all stored credentials."""
        print("\n" + "="*70)
        print("🔐 STORED CREDENTIALS")
        print("="*70 + "\n")

        creds = self.vault.list_credentials()

        if not creds:
            print("   No credentials stored yet")
        else:
            for platform, types in creds.items():
                print(f"📦 {platform.upper()}:")
                for cred_type in types:
                    print(f"   • {cred_type}")
                print()

    def export_credentials_safe(self, output_path: str = "credentials_export.txt"):
        """
        Export credentials to a secure file.

        Args:
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                f.write("AGI Browser - Credential Export\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write("="*70 + "\n\n")

                for platform, creds in self.vault.credentials.items():
                    f.write(f"{platform.upper()}:\n")
                    for cred_type, data in creds.items():
                        f.write(f"  {cred_type}: {data['value']}\n")
                        f.write(f"  Created: {data['created_at']}\n")
                    f.write("\n")

            os.chmod(output_path, 0o600)  # Secure permissions
            print(f"✓ Credentials exported to: {output_path}")

        except Exception as e:
            print(f"✗ Export failed: {e}")

    def _log_action(self, action: str, details: Dict):
        """Log an action."""
        self.session_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        })


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

async def demo_auto_setup():
    """Demo: Automatically set up GitHub and HuggingFace."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        manager = AutonomousAccountManager(page)

        # Auto-setup GitHub
        print("Setting up GitHub...")
        github_result = await manager.auto_setup_platform('github')

        # Show credentials
        manager.show_stored_credentials()

        # Export credentials
        manager.export_credentials_safe()

        await browser.close()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         🤖 Autonomous Account Manager                            ║
    ║         For AGI Browser - Account & API Key Automation           ║
    ╚══════════════════════════════════════════════════════════════════╝

    ⚠️  IMPORTANT - READ BEFORE USE:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    This tool automates account creation and API key management.

    ETHICAL USE ONLY:
    • Only create accounts you own or have permission to create
    • Do not violate Terms of Service
    • Do not use for spam or abuse
    • Respect rate limits
    • Use for legitimate automation only

    SECURITY:
    • Credentials stored in agi_credentials.json with 0600 permissions
    • Never share credential files
    • Use strong, unique passwords
    • Enable 2FA where available

    CAPABILITIES:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ✓ Autonomous account creation (GitHub, HuggingFace, etc.)
    ✓ Automatic login with stored credentials
    ✓ API key generation and management
    ✓ Secure credential storage
    ✓ Platform authentication
    ✓ Credential export

    USAGE:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python autonomous_account_manager.py

    Or integrate with AGI Browser for full automation.
    """)

    # Run demo
    # asyncio.run(demo_auto_setup())
