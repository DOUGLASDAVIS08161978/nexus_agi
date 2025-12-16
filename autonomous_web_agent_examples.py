"""
NEXUS AGI - Autonomous Web Agent Examples
Advanced usage examples and scenarios
"""

from autonomous_web_agent import (
    AutonomousWebAgent,
    SecureCredentialVault,
    WebBrowser,
    AccountCreationAgent
)
import json


def example_1_credential_generation():
    """Example 1: Generate credentials for multiple services."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Credential Generation for Multiple Services")
    print("="*60 + "\n")

    vault = SecureCredentialVault("example_vault.json")

    services = [
        "GitHub",
        "GitLab",
        "Bitbucket",
        "Docker Hub",
        "NPM Registry",
        "PyPI",
        "AWS",
        "Google Cloud",
        "Azure"
    ]

    for service in services:
        creds = vault.create_account_credentials(service)
        print(f"✅ {service:20s} | User: {creds['username']:25s} | API Key: {creds['api_key'][:30]}...")

    print(f"\n💾 Total credentials stored: {len(vault.credentials)}")


def example_2_api_key_management():
    """Example 2: API key generation and management."""
    print("\n" + "="*60)
    print("EXAMPLE 2: API Key Management")
    print("="*60 + "\n")

    vault = SecureCredentialVault("api_keys_vault.json")

    # Generate different types of API keys
    api_configs = [
        ("github", {"scope": "repo,user", "rate_limit": 5000}),
        ("openai", {"model": "gpt-4", "tier": "premium"}),
        ("stripe", {"mode": "live", "permissions": "full"}),
        ("aws", {"region": "us-east-1", "services": ["s3", "ec2"]}),
        ("google", {"apis": ["maps", "translate", "vision"]})
    ]

    for prefix, metadata in api_configs:
        api_key = vault.generate_api_key(prefix)
        vault.store_api_key(prefix, api_key, metadata)
        print(f"🔑 {prefix.upper():10s} API Key: {api_key[:40]}...")
        print(f"   Metadata: {json.dumps(metadata, indent=11)[11:]}\n")


def example_3_token_lifecycle():
    """Example 3: Token generation and lifecycle management."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Token Lifecycle Management")
    print("="*60 + "\n")

    vault = SecureCredentialVault("tokens_vault.json")

    # Generate various token types
    token_configs = [
        ("bearer", "GitHub", 3600),
        ("access", "Google OAuth", 7200),
        ("refresh", "Google OAuth", 86400),
        ("jwt", "Internal API", 1800),
        ("session", "Web App", 43200)
    ]

    for token_type, service, expiry in token_configs:
        token = vault.generate_token(token_type)
        vault.store_token(service, token, token_type, expiry)
        print(f"🎫 {token_type.upper():10s} for {service:20s}")
        print(f"   Token: {token[:50]}...")
        print(f"   Expires in: {expiry}s ({expiry/3600:.1f} hours)\n")


def example_4_web_research_automation():
    """Example 4: Autonomous web research."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Autonomous Web Research")
    print("="*60 + "\n")

    agent = AutonomousWebAgent()

    research_topics = [
        "Large Language Models 2025",
        "Quantum Computing Breakthroughs",
        "AGI Safety Research",
        "Neural Network Architectures",
        "Reinforcement Learning Applications"
    ]

    all_research = []

    for topic in research_topics:
        print(f"\n📚 Researching: {topic}")
        research = agent.autonomous_research(topic, depth=2)
        all_research.append(research)
        print(f"   ✓ Found {len(research['key_findings'])} key findings")

    # Generate research summary
    print("\n" + "="*60)
    print("RESEARCH SUMMARY")
    print("="*60)
    print(f"\nTotal topics researched: {len(all_research)}")
    print(f"Total sources analyzed: {sum(len(r['sources']) for r in all_research)}")
    print(f"Total findings extracted: {sum(len(r['key_findings']) for r in all_research)}")


def example_5_multi_service_registration():
    """Example 5: Register on multiple services autonomously."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Multi-Service Registration")
    print("="*60 + "\n")

    agent = AutonomousWebAgent()

    services_to_register = [
        ("GitHub", "https://github.com/signup"),
        ("GitLab", "https://gitlab.com/users/sign_up"),
        ("Bitbucket", "https://bitbucket.org/account/signup/"),
        ("Heroku", "https://signup.heroku.com/"),
        ("Vercel", "https://vercel.com/signup"),
        ("Netlify", "https://app.netlify.com/signup"),
        ("Railway", "https://railway.app/signup"),
        ("Render", "https://dashboard.render.com/register")
    ]

    successful = []
    failed = []

    for service_name, signup_url in services_to_register:
        print(f"\n🔧 Registering on {service_name}...")
        result = agent.register_on_service(service_name, signup_url)

        if result["success"]:
            successful.append(service_name)
            print(f"   ✅ Success!")
        else:
            failed.append(service_name)
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

    print("\n" + "="*60)
    print("REGISTRATION SUMMARY")
    print("="*60)
    print(f"\n✅ Successful: {len(successful)}/{len(services_to_register)}")
    print(f"❌ Failed: {len(failed)}/{len(services_to_register)}")

    if successful:
        print(f"\nSuccessfully registered on: {', '.join(successful)}")


def example_6_api_interaction_workflow():
    """Example 6: Complete API interaction workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 6: API Interaction Workflow")
    print("="*60 + "\n")

    agent = AutonomousWebAgent()

    # Workflow: Register -> Authenticate -> Use API
    service_name = "Example API Service"
    service_url = "https://api.example.com/signup"

    # Step 1: Register
    print("Step 1: Registration")
    print("-" * 40)
    reg_result = agent.register_on_service(service_name, service_url)

    if reg_result["success"]:
        # Step 2: Authenticate
        print("\nStep 2: Authentication")
        print("-" * 40)
        auth_result = agent.account_agent.authenticate(
            service_name,
            "https://api.example.com/login"
        )

        if auth_result["success"]:
            # Step 3: Use API endpoints
            print("\nStep 3: API Operations")
            print("-" * 40)

            api_operations = [
                ("https://api.example.com/user/profile", "get_profile"),
                ("https://api.example.com/user/settings", "get_settings"),
                ("https://api.example.com/data/list", "list_data"),
                ("https://api.example.com/analytics", "get_analytics")
            ]

            for endpoint, action in api_operations:
                result = agent.interact_with_api(service_name, endpoint, action)
                print(f"   ✓ {action}")

    # Generate workflow report
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    status = agent.get_status_report()
    print(f"\nTotal API calls made: {len([t for t in status['tasks'] if t['task'] == 'api_interaction'])}")


def example_7_secure_password_generation():
    """Example 7: Demonstrate secure password generation patterns."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Secure Password Generation Patterns")
    print("="*60 + "\n")

    vault = SecureCredentialVault()

    password_configs = [
        (12, "Minimum security"),
        (16, "Standard security"),
        (24, "High security"),
        (32, "Maximum security"),
        (64, "Paranoid security")
    ]

    for length, description in password_configs:
        password = vault.generate_password(length)

        # Analyze password complexity
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        print(f"Length: {length:2d} | {description:20s}")
        print(f"   Password: {password}")
        print(f"   Complexity: Upper={has_upper}, Lower={has_lower}, Digits={has_digit}, Special={has_special}")
        print()


def example_8_credential_vault_operations():
    """Example 8: Advanced credential vault operations."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Advanced Credential Vault Operations")
    print("="*60 + "\n")

    vault = SecureCredentialVault("advanced_vault.json")

    # Create accounts with custom usernames
    print("Creating custom accounts:")
    custom_accounts = [
        ("GitHub", "agi_developer"),
        ("GitLab", "agi_researcher"),
        ("DockerHub", "agi_containerizer")
    ]

    for service, username in custom_accounts:
        creds = vault.create_account_credentials(service, username)
        print(f"   ✓ {service}: {username} | Email: {creds['email']}")

    # Store API keys with metadata
    print("\nStoring API keys with metadata:")
    vault.store_api_key("Stripe", vault.generate_api_key("stripe"), {
        "environment": "production",
        "permissions": ["charges:write", "customers:read"],
        "webhook_secret": "whsec_" + vault.generate_api_key()[:32]
    })
    print("   ✓ Stripe API key stored with metadata")

    # Store tokens
    print("\nStoring authentication tokens:")
    vault.store_token("GitHub", vault.generate_token("bearer"), "bearer", 7200)
    vault.store_token("GitLab", vault.generate_token("pat"), "personal_access", None)
    print("   ✓ Multiple tokens stored")

    # Retrieve and display
    print("\n" + "-"*60)
    print("Vault Contents:")
    print("-"*60)
    print(f"Credentials: {len(vault.credentials)}")
    print(f"API Keys: {len(vault.api_keys)}")
    print(f"Tokens: {len(vault.tokens)}")

    # Display detailed credentials
    for service, creds in vault.credentials.items():
        print(f"\n{service}:")
        print(f"   Username: {creds['username']}")
        print(f"   Created: {creds['created_at']}")
        print(f"   Last Used: {creds['last_used']}")


def example_9_browser_history_tracking():
    """Example 9: Browser history and activity tracking."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Browser History and Activity Tracking")
    print("="*60 + "\n")

    browser = WebBrowser(user_agent="NEXUS-AGI-Research/2.0")

    # Simulate browsing activity
    queries = [
        "machine learning frameworks",
        "neural network optimization",
        "transformer architecture",
        "attention mechanisms"
    ]

    for query in queries:
        results = browser.search(query)
        print(f"🔍 Searched: {query} ({len(results)} results)")

        # Visit top result
        if results:
            browser.fetch_page(results[0]["url"])

    # Display history
    print("\n" + "-"*60)
    print("Browsing History:")
    print("-"*60)

    for i, entry in enumerate(browser.history, 1):
        print(f"{i}. {entry['action'].upper()}")
        if entry['action'] == 'search':
            print(f"   Query: {entry['query']}")
            print(f"   Results: {entry['results_count']}")
        elif entry['action'] == 'fetch':
            print(f"   URL: {entry['url']}")
            print(f"   Status: {entry['status']}")
        print(f"   Time: {entry['timestamp']}\n")


def run_all_examples():
    """Run all examples in sequence."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     NEXUS AGI - AUTONOMOUS WEB AGENT EXAMPLES                ║
║                                                              ║
║  Demonstrating Advanced Autonomous Web Capabilities          ║
╚══════════════════════════════════════════════════════════════╝
    """)

    examples = [
        ("Credential Generation", example_1_credential_generation),
        ("API Key Management", example_2_api_key_management),
        ("Token Lifecycle", example_3_token_lifecycle),
        ("Web Research Automation", example_4_web_research_automation),
        ("Multi-Service Registration", example_5_multi_service_registration),
        ("API Interaction Workflow", example_6_api_interaction_workflow),
        ("Secure Password Generation", example_7_secure_password_generation),
        ("Credential Vault Operations", example_8_credential_vault_operations),
        ("Browser History Tracking", example_9_browser_history_tracking)
    ]

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n\n{'#'*60}")
        print(f"# RUNNING EXAMPLE {i}/{len(examples)}: {name.upper()}")
        print(f"{'#'*60}")

        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")

        input("\nPress Enter to continue to next example...")

    print("""
╔══════════════════════════════════════════════════════════════╗
║              ALL EXAMPLES COMPLETED                          ║
║                                                              ║
║  The autonomous web agent is ready for production use!       ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        examples_map = {
            "1": example_1_credential_generation,
            "2": example_2_api_key_management,
            "3": example_3_token_lifecycle,
            "4": example_4_web_research_automation,
            "5": example_5_multi_service_registration,
            "6": example_6_api_interaction_workflow,
            "7": example_7_secure_password_generation,
            "8": example_8_credential_vault_operations,
            "9": example_9_browser_history_tracking
        }

        if example_num in examples_map:
            examples_map[example_num]()
        else:
            print(f"Unknown example: {example_num}")
            print("Available examples: 1-9")
    else:
        # Run specific examples without interaction
        print("Running selected examples...\n")
        example_1_credential_generation()
        example_2_api_key_management()
        example_7_secure_password_generation()
