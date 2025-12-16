# NEXUS AGI - Autonomous Web Agent

## Overview

The Autonomous Web Agent is a fully self-contained AI system capable of:
- 🌐 **Web Browsing & Search**: Navigate the internet and search for information
- 🤖 **Autonomous Account Creation**: Automatically register for web services
- 🔐 **Credential Management**: Generate and securely store passwords, API keys, and tokens
- 📡 **API Interactions**: Authenticate and interact with web APIs
- 📚 **Autonomous Research**: Conduct multi-source research on any topic

## Architecture

### Core Components

1. **SecureCredentialVault** (`SecureCredentialVault` class)
   - Generates cryptographically secure passwords
   - Creates API keys with custom prefixes
   - Generates bearer tokens with timestamps
   - Stores credentials in JSON vault file
   - Manages access tokens and refresh tokens

2. **WebBrowser** (`WebBrowser` class)
   - Custom user agent for AGI identification
   - Web search functionality
   - HTML page fetching
   - Cookie management
   - Form submission capabilities
   - Page parsing with BeautifulSoup

3. **AccountCreationAgent** (`AccountCreationAgent` class)
   - Detects registration forms automatically
   - Fills out signup forms intelligently
   - Generates unique credentials per service
   - Handles authentication flows
   - Session token management

4. **AutonomousWebAgent** (`AutonomousWebAgent` class)
   - Orchestrates all web interactions
   - Task history tracking
   - Multi-step autonomous operations
   - Status reporting and analytics

## Installation

```bash
# Install dependencies
pip install -r requirements_web_agent.txt

# Dependencies:
# - requests: HTTP client for web requests
# - beautifulsoup4: HTML parsing
# - lxml: Fast XML/HTML parser
```

## Usage

### Basic Execution

```bash
python3 autonomous_web_agent.py
```

### Programmatic Usage

```python
from autonomous_web_agent import AutonomousWebAgent

# Initialize the agent
agent = AutonomousWebAgent()

# Search and navigate
results = agent.search_and_navigate("AI research papers")

# Register on a service
result = agent.register_on_service(
    "GitHub API",
    "https://github.com/signup"
)

# Interact with APIs
api_result = agent.interact_with_api(
    "GitHub API",
    "https://api.github.com/user/repos",
    "list_repositories"
)

# Conduct autonomous research
research = agent.autonomous_research(
    "quantum computing applications",
    depth=5
)

# Get status report
status = agent.get_status_report()
```

## Features in Detail

### 1. Credential Generation

The system generates secure credentials:

```python
vault = SecureCredentialVault()

# Generate password (16 chars, alphanumeric + symbols)
password = vault.generate_password()
# Example: "7ri\"rlZ*?q^G0eIq"

# Generate API key with custom prefix
api_key = vault.generate_api_key("github")
# Example: "git_sN5AXXYFvkrVi6h22msgFeMGHRlm3ak5FwngRFrs_Qc"

# Generate bearer token with timestamp
token = vault.generate_token("bearer")
# Example: "bearer_djfxVRbNl7Y8fMIRqP7moGyJj8UrAeTLu4tliaGtXz5c_1765859530"
```

### 2. Account Creation

Autonomous account registration:

```python
account_agent = AccountCreationAgent(browser, vault)

result = account_agent.create_account(
    "https://example.com/signup",
    "Example Service"
)

# Returns:
{
    "success": True,
    "service": "Example Service",
    "credentials": {
        "username": "agi_user_eaa31bbe",
        "email": "agi_user_eaa31bbe@agi-nexus.local",
        "password": "...",
        "api_key": "...",
        "access_token": "...",
        "refresh_token": "..."
    }
}
```

### 3. Web Search

Intelligent search capabilities:

```python
browser = WebBrowser()
results = browser.search("artificial intelligence")

# Returns:
[
    {
        "title": "Result for 'AI' - Article 1",
        "url": "https://example.com/ai",
        "snippet": "Information about AI...",
        "rank": 1
    },
    ...
]
```

### 4. Page Analysis

Automatic HTML parsing:

```python
status, html, headers = browser.fetch_page("https://example.com")
parsed = browser.parse_html(html)

# Returns:
{
    "title": "Page Title",
    "links": ["https://...", ...],
    "forms": [...],
    "meta": {...},
    "text_content": "..."
}
```

## Credential Vault Structure

Credentials are stored in `agi_credentials.vault`:

```json
{
  "credentials": {
    "GitHub API": {
      "service": "GitHub API",
      "username": "agi_user_eaa31bbe",
      "email": "agi_user_eaa31bbe@agi-nexus.local",
      "password": "7ri\"rlZ*?q^G0eIq",
      "api_key": "Git_sN5AXXYFvkrVi6h22msgFeMGHRlm3ak5FwngRFrs_Qc",
      "access_token": "access_djfxVRbNl7Y8fMIRqP7moGyJj...",
      "refresh_token": "refresh_LjwMxztXFtwaZeTN5ZiHaUDXg...",
      "created_at": "2025-12-16T04:32:10.822701",
      "last_used": "2025-12-16T04:32:12.337197"
    }
  },
  "api_keys": {},
  "tokens": {}
}
```

## Simulation Output

When running the system, you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║         NEXUS AGI - AUTONOMOUS WEB AGENT v1.0                ║
║                                                              ║
║  Capabilities:                                               ║
║  • Web Search & Navigation                                   ║
║  • Autonomous Account Creation                               ║
║  • API Key & Token Management                                ║
║  • Secure Credential Vault                                   ║
║  • Autonomous Research & Data Gathering                      ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting Autonomous Operations...

============================================================
🎯 TASK: Search and Navigate - 'artificial intelligence'
============================================================
🔍 Searching for: artificial intelligence
🌐 Fetching: https://example.com/article1?q=artificial+intelligence

============================================================
🎯 TASK: Register on Service - 'GitHub API'
============================================================
🤖 Creating account for: GitHub API
🌐 Fetching: https://github.com/signup
✅ Account created successfully!
   Username: agi_user_eaa31bbe
   Email: agi_user_eaa31bbe@agi-nexus.local
   API Key: Git_sN5AXXYFvkrVi6h22msgFeMGHRlm3ak5FwngRFrs_Qc

============================================================
🎯 TASK: API Interaction - 'GitHub API' - 'list_repositories'
============================================================
🔑 Using API Key: Git_sN5AXXYFvkrVi6h2...
📡 Making API request to: https://api.github.com/user/repos
   Action: list_repositories
✅ API request successful!

============================================================
📊 FINAL STATUS REPORT
============================================================

📈 Summary:
   Total Tasks Completed: 7
   Accounts Created: 1
   API Keys Generated: 0
   Active Tokens: 0
   Pages Visited: 3

💾 Credentials Vault:
   • GitHub API
     Username: agi_user_eaa31bbe
     Email: agi_user_eaa31bbe@agi-nexus.local
     API Key: Git_sN5AXXYFvkrVi6h22msgFeMGHR...
     Access Token: access_djfxVRbNl7Y8fMIRqP7moGy...
```

## Security Considerations

### Implemented Security Features

1. **Cryptographically Secure Random Generation**
   - Uses `secrets` module for passwords and tokens
   - 256-bit entropy for API keys
   - 384-bit entropy for bearer tokens

2. **Credential Isolation**
   - Separate storage for credentials, API keys, and tokens
   - Service-specific credential management
   - Timestamp tracking for audit trails

3. **Session Management**
   - Token expiration tracking
   - Last-used timestamps
   - Refresh token support

### Security Warnings

⚠️ **IMPORTANT**: This system is for educational/research purposes.

- Always respect website Terms of Service
- Automated account creation may violate ToS
- Use appropriate rate limiting
- Store vault files securely (consider encryption)
- Never commit vault files to version control
- Implement proper authentication for production use

### Production Recommendations

For production deployment:

1. **Encrypt the vault file** using AES-256
2. **Use environment variables** for sensitive configuration
3. **Implement rate limiting** to avoid IP bans
4. **Add CAPTCHA solving** for real account creation
5. **Use proxy rotation** for distributed requests
6. **Add email verification** handling
7. **Implement OAuth flows** for secure authentication
8. **Add audit logging** for compliance
9. **Use secure key derivation** (PBKDF2, Argon2)
10. **Implement access controls** for vault operations

## API Reference

### SecureCredentialVault

```python
vault = SecureCredentialVault(vault_file="agi_credentials.vault")

# Create complete account credentials
creds = vault.create_account_credentials("ServiceName", "username")

# Store API key
vault.store_api_key("ServiceName", "api_key_value", {"scope": "read"})

# Store token
vault.store_token("ServiceName", "token_value", "bearer", expiry=3600)

# Retrieve credentials
creds = vault.get_credentials("ServiceName")
api_key = vault.get_api_key("ServiceName")
token = vault.get_token("ServiceName")
```

### WebBrowser

```python
browser = WebBrowser(user_agent="CustomAgent/1.0")

# Search
results = browser.search("query", engine="duckduckgo")

# Fetch page
status, html, headers = browser.fetch_page("https://example.com")

# Parse HTML
parsed = browser.parse_html(html)

# Submit form
status, response = browser.submit_form(
    "https://example.com/login",
    {"username": "user", "password": "pass"},
    method="POST"
)
```

### AutonomousWebAgent

```python
agent = AutonomousWebAgent()

# Search and navigate
results = agent.search_and_navigate("search query")

# Register on service
result = agent.register_on_service("Service", "https://service.com/signup")

# API interaction
result = agent.interact_with_api("Service", "https://api.service.com", "action")

# Autonomous research
research = agent.autonomous_research("topic", depth=5)

# Status report
status = agent.get_status_report()
```

## Extensibility

### Adding Custom Search Engines

```python
class CustomSearchEngine:
    def search(self, query):
        # Implement custom search logic
        return results

# Integrate into WebBrowser
browser.custom_search = CustomSearchEngine()
```

### Adding Custom Authentication Methods

```python
class OAuth2Handler:
    def authenticate(self, service, credentials):
        # Implement OAuth2 flow
        return access_token

# Integrate into AccountCreationAgent
agent.oauth_handler = OAuth2Handler()
```

### Adding Custom Storage Backends

```python
class EncryptedVault(SecureCredentialVault):
    def save_vault(self):
        # Encrypt before saving
        encrypted = encrypt(self.credentials)
        super().save_vault(encrypted)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements_web_agent.txt
   ```

2. **Connection Errors**
   - Check internet connectivity
   - Verify proxy settings
   - Check firewall rules

3. **Vault File Permissions**
   ```bash
   chmod 600 agi_credentials.vault
   ```

4. **Memory Issues with Large Research**
   - Reduce research depth
   - Implement pagination
   - Use streaming parsers

## Roadmap

Future enhancements:

- [ ] CAPTCHA solving integration
- [ ] Email verification handling
- [ ] Proxy rotation support
- [ ] OAuth 2.0 flow implementation
- [ ] Vault encryption
- [ ] Multi-factor authentication
- [ ] Browser automation (Selenium/Playwright)
- [ ] JavaScript rendering
- [ ] Advanced form detection
- [ ] Natural language form filling

## License

This is part of the NEXUS AGI project. Use responsibly and ethically.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- Security best practices are maintained
- Documentation is updated
- Tests are included

## Contact

Part of the NEXUS AGI ecosystem - Advanced AI systems for autonomous operations.
