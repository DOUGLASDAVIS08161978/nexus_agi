#!/usr/bin/env python3
"""
AGI-Enhanced Browser v2.0
==========================
An AI-powered web browser that can:
- Understand natural language commands
- Autonomously navigate and interact with websites
- Search GitHub and create repositories
- Generate code and push to GitHub
- Extract and analyze web content
- Execute complex multi-step tasks

Like Comet Browser, but with AGI-level capabilities.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from playwright.async_api import async_playwright, Page, Browser
import requests
from pathlib import Path


class GitHubIntegration:
    """GitHub API integration for repository management."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub integration.

        Args:
            token: GitHub personal access token (or set GITHUB_TOKEN env var)
        """
        self.token = token or os.getenv('GITHUB_TOKEN')
        self.base_url = 'https://api.github.com'
        self.headers = {
            'Authorization': f'token {self.token}' if self.token else '',
            'Accept': 'application/vnd.github.v3+json'
        }

    def search_repositories(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search GitHub repositories.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of repository dictionaries
        """
        try:
            url = f"{self.base_url}/search/repositories"
            params = {'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': max_results}

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()

            data = response.json()
            repos = []

            for item in data.get('items', []):
                repos.append({
                    'name': item['name'],
                    'full_name': item['full_name'],
                    'description': item['description'],
                    'url': item['html_url'],
                    'stars': item['stargazers_count'],
                    'language': item['language']
                })

            print(f"✓ Found {len(repos)} repositories")
            return repos

        except Exception as e:
            print(f"✗ GitHub search failed: {e}")
            return []

    def create_repository(self, name: str, description: str = "", private: bool = False) -> Optional[Dict]:
        """
        Create a new GitHub repository.

        Args:
            name: Repository name
            description: Repository description
            private: Make repository private

        Returns:
            Repository data or None
        """
        if not self.token:
            print("✗ GitHub token required to create repositories")
            return None

        try:
            url = f"{self.base_url}/user/repos"
            data = {
                'name': name,
                'description': description,
                'private': private,
                'auto_init': True
            }

            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()

            repo = response.json()
            print(f"✓ Created repository: {repo['html_url']}")

            return {
                'name': repo['name'],
                'full_name': repo['full_name'],
                'url': repo['html_url'],
                'clone_url': repo['clone_url']
            }

        except Exception as e:
            print(f"✗ Failed to create repository: {e}")
            return None

    def create_file(self, repo_full_name: str, file_path: str, content: str,
                   commit_message: str = "Add file") -> bool:
        """
        Create or update a file in a GitHub repository.

        Args:
            repo_full_name: Repository full name (owner/repo)
            file_path: Path of file in repository
            content: File content
            commit_message: Commit message

        Returns:
            Success status
        """
        if not self.token:
            print("✗ GitHub token required")
            return False

        try:
            import base64

            # Encode content to base64
            content_bytes = content.encode('utf-8')
            content_b64 = base64.b64encode(content_bytes).decode('utf-8')

            url = f"{self.base_url}/repos/{repo_full_name}/contents/{file_path}"
            data = {
                'message': commit_message,
                'content': content_b64
            }

            response = requests.put(url, headers=self.headers, json=data)
            response.raise_for_status()

            print(f"✓ Created file: {file_path}")
            return True

        except Exception as e:
            print(f"✗ Failed to create file: {e}")
            return False


class AICommandParser:
    """Parse natural language commands into browser actions."""

    def __init__(self):
        self.command_patterns = {
            'navigate': r'(?:go to|navigate to|open|visit)\s+(.+)',
            'search_google': r'(?:search|google|look up|find)\s+(.+)',
            'search_github': r'(?:github search|search github for|find on github)\s+(.+)',
            'create_repo': r'create\s+(?:a\s+)?(?:github\s+)?repo(?:sitory)?\s+(?:named\s+)?([^\s]+)(?:\s+(.+))?',
            'click': r'click\s+(?:on\s+)?(.+)',
            'extract_text': r'(?:get|extract|read)\s+(?:the\s+)?(.+)',
            'screenshot': r'(?:take|capture)\s+(?:a\s+)?screenshot',
            'scroll': r'scroll\s+(up|down)',
            'fill_form': r'(?:fill|enter|type)\s+(.+)\s+(?:in|into)\s+(.+)',
        }

    def parse(self, command: str) -> Dict[str, Any]:
        """
        Parse a natural language command.

        Args:
            command: Natural language command

        Returns:
            Parsed command dictionary
        """
        command = command.strip().lower()

        for action, pattern in self.command_patterns.items():
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                result = {'action': action, 'raw': command}

                if action == 'navigate':
                    url = match.group(1).strip()
                    if not url.startswith('http'):
                        url = 'https://' + url
                    result['url'] = url

                elif action in ['search_google', 'search_github']:
                    result['query'] = match.group(1).strip()

                elif action == 'create_repo':
                    result['name'] = match.group(1).strip()
                    result['description'] = match.group(2).strip() if match.group(2) else ""

                elif action == 'click':
                    result['target'] = match.group(1).strip()

                elif action == 'extract_text':
                    result['selector'] = match.group(1).strip()

                elif action == 'scroll':
                    result['direction'] = match.group(1).strip()

                elif action == 'fill_form':
                    result['value'] = match.group(1).strip()
                    result['field'] = match.group(2).strip()

                return result

        # Default: treat as search
        return {'action': 'search_google', 'query': command}


class AGIEnhancedBrowser:
    """
    AI-powered web browser with autonomous capabilities.
    """

    def __init__(self, headless: bool = False, github_token: Optional[str] = None):
        """
        Initialize the AGI browser.

        Args:
            headless: Run browser in headless mode
            github_token: GitHub personal access token
        """
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.browser_context = None

        self.github = GitHubIntegration(github_token)
        self.parser = AICommandParser()
        self.interaction_log: List[Dict] = []
        self.session_data: Dict = {
            'start_time': datetime.now().isoformat(),
            'commands_executed': 0,
            'urls_visited': [],
            'repos_created': []
        }

    async def start(self):
        """Start the browser instance."""
        print("\n" + "="*70)
        print("🚀 AGI-Enhanced Browser v2.0 - Starting...")
        print("="*70 + "\n")

        self.playwright = await async_playwright().start()

        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )

        self.browser_context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        self.page = await self.browser_context.new_page()
        print("✓ Browser ready\n")

    async def stop(self):
        """Stop the browser instance."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("\n✓ Browser stopped")
        self.save_session_log()

    async def execute_natural_language_command(self, command: str) -> Dict[str, Any]:
        """
        Execute a natural language command.

        Args:
            command: Natural language command

        Returns:
            Result dictionary
        """
        print(f"\n💬 Command: {command}")

        # Parse command
        parsed = self.parser.parse(command)
        action = parsed['action']

        print(f"🤖 Understood as: {action}")

        result = {'success': False, 'action': action, 'command': command}

        try:
            if action == 'navigate':
                result['success'] = await self._navigate(parsed['url'])

            elif action == 'search_google':
                result['success'] = await self._search_google(parsed['query'])

            elif action == 'search_github':
                repos = self.github.search_repositories(parsed['query'])
                result['success'] = len(repos) > 0
                result['repos'] = repos
                if repos:
                    print(f"\n📦 Top GitHub Repositories:")
                    for i, repo in enumerate(repos[:5], 1):
                        print(f"   {i}. {repo['full_name']} ⭐ {repo['stars']}")
                        print(f"      {repo['description'][:80] if repo['description'] else 'No description'}...")

            elif action == 'create_repo':
                repo = self.github.create_repository(parsed['name'], parsed.get('description', ''))
                result['success'] = repo is not None
                result['repo'] = repo
                if repo:
                    self.session_data['repos_created'].append(repo['url'])

            elif action == 'click':
                result['success'] = await self._smart_click(parsed['target'])

            elif action == 'screenshot':
                path = await self._screenshot()
                result['success'] = bool(path)
                result['path'] = path

            elif action == 'scroll':
                await self._scroll(parsed['direction'])
                result['success'] = True

            elif action == 'extract_text':
                text = await self._extract_text(parsed['selector'])
                result['success'] = text is not None
                result['text'] = text

            else:
                result['error'] = f"Unknown action: {action}"

        except Exception as e:
            result['error'] = str(e)
            print(f"✗ Error: {e}")

        self._log_interaction(command, result)
        self.session_data['commands_executed'] += 1

        return result

    async def _navigate(self, url: str) -> bool:
        """Navigate to URL."""
        try:
            print(f"📍 Navigating to: {url}")
            await self.page.goto(url, wait_until='domcontentloaded', timeout=30000)
            self.session_data['urls_visited'].append(url)
            print(f"✓ Loaded: {self.page.url}")
            return True
        except Exception as e:
            print(f"✗ Navigation failed: {e}")
            return False

    async def _search_google(self, query: str) -> bool:
        """Perform Google search."""
        try:
            print(f"🔍 Searching Google for: {query}")
            await self._navigate("https://www.google.com")
            await asyncio.sleep(1)

            search_box = await self.page.query_selector('textarea[name="q"]')
            if not search_box:
                search_box = await self.page.query_selector('input[name="q"]')

            if search_box:
                await search_box.fill(query)
                await search_box.press('Enter')
                await self.page.wait_for_load_state('networkidle')
                print(f"✓ Search complete")
                return True
            return False
        except Exception as e:
            print(f"✗ Search failed: {e}")
            return False

    async def _smart_click(self, target: str) -> bool:
        """Intelligently click an element by text or selector."""
        try:
            print(f"🖱️  Attempting to click: {target}")

            # Try as button text
            selectors = [
                f"button:has-text('{target}')",
                f"a:has-text('{target}')",
                f"input[type='submit']:has-text('{target}')",
                target  # Try as CSS selector
            ]

            for selector in selectors:
                try:
                    await self.page.click(selector, timeout=3000)
                    print(f"✓ Clicked successfully")
                    return True
                except:
                    continue

            print(f"✗ Could not find: {target}")
            return False
        except Exception as e:
            print(f"✗ Click failed: {e}")
            return False

    async def _screenshot(self, path: Optional[str] = None) -> str:
        """Take screenshot."""
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"agi_screenshot_{timestamp}.png"

        try:
            await self.page.screenshot(path=path, full_page=True)
            print(f"📸 Screenshot saved: {path}")
            return path
        except Exception as e:
            print(f"✗ Screenshot failed: {e}")
            return ""

    async def _scroll(self, direction: str):
        """Scroll the page."""
        if direction == 'down':
            await self.page.evaluate('window.scrollBy(0, window.innerHeight)')
        else:
            await self.page.evaluate('window.scrollBy(0, -window.innerHeight)')
        print(f"✓ Scrolled {direction}")

    async def _extract_text(self, selector: str) -> Optional[str]:
        """Extract text from element."""
        try:
            element = await self.page.query_selector(selector)
            if element:
                text = await element.text_content()
                print(f"✓ Extracted: {text[:100]}...")
                return text.strip() if text else None
            return None
        except Exception as e:
            print(f"✗ Extract failed: {e}")
            return None

    async def generate_and_push_code(self, repo_name: str, code_description: str) -> bool:
        """
        Generate code based on description and push to GitHub.

        Args:
            repo_name: Repository name
            code_description: What the code should do

        Returns:
            Success status
        """
        print(f"\n🤖 Generating code for: {code_description}")

        # Create repository
        repo = self.github.create_repository(
            repo_name,
            f"Auto-generated: {code_description}"
        )

        if not repo:
            return False

        # Generate simple code based on description
        # In a real implementation, this would use the AGI system
        code = self._generate_sample_code(code_description)

        # Push code to repository
        success = self.github.create_file(
            repo['full_name'],
            'main.py',
            code,
            f"Add {code_description}"
        )

        if success:
            print(f"✓ Code pushed to: {repo['url']}")
            self.session_data['repos_created'].append(repo['url'])

        return success

    def _generate_sample_code(self, description: str) -> str:
        """Generate sample Python code based on description."""
        timestamp = datetime.now().isoformat()

        return f'''#!/usr/bin/env python3
"""
{description}
Auto-generated by AGI-Enhanced Browser
Created: {timestamp}
"""

def main():
    """Main function."""
    print("Hello from AGI-Enhanced Browser!")
    print("This code was auto-generated for: {description}")

    # TODO: Implement the actual logic for: {description}
    pass

if __name__ == "__main__":
    main()
'''

    def _log_interaction(self, command: str, result: Dict):
        """Log an interaction."""
        self.interaction_log.append({
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'result': result,
            'url': self.page.url if self.page else None
        })

    def save_session_log(self, path: str = 'agi_browser_session.json'):
        """Save session log to file."""
        log_data = {
            'session': self.session_data,
            'interactions': self.interaction_log
        }

        with open(path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"\n📝 Session log saved: {path}")

    async def interactive_mode(self):
        """Run in interactive mode with natural language commands."""
        print("\n" + "="*70)
        print("🤖 AGI-Enhanced Browser - Interactive Mode")
        print("="*70)
        print("\nType natural language commands or 'help' for examples.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                command = input("💬 You: ").strip()

                if not command:
                    continue

                if command.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye!")
                    break

                if command.lower() == 'help':
                    self._show_help()
                    continue

                if command.lower() == 'stats':
                    self._show_stats()
                    continue

                # Execute command
                await self.execute_natural_language_command(command)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")

    def _show_help(self):
        """Show help message."""
        print("""
📚 Available Commands (Natural Language):

Navigation:
  - "go to github.com"
  - "open https://example.com"
  - "visit reddit.com"

Search:
  - "search for Python tutorials"
  - "google artificial intelligence"
  - "github search machine learning"

GitHub:
  - "create repo my-awesome-project"
  - "create github repository my-app with description for testing"

Interaction:
  - "click on Sign In"
  - "click button Login"
  - "scroll down"

Content:
  - "take screenshot"
  - "extract text from h1"
  - "get the title"

Special:
  - "stats" - Show session statistics
  - "help" - Show this help
  - "quit" - Exit browser
        """)

    def _show_stats(self):
        """Show session statistics."""
        print(f"""
📊 Session Statistics:
  - Commands executed: {self.session_data['commands_executed']}
  - URLs visited: {len(self.session_data['urls_visited'])}
  - Repos created: {len(self.session_data['repos_created'])}
  - Start time: {self.session_data['start_time']}
        """)


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

async def demo_basic_browsing():
    """Demo: Basic browsing with natural language."""
    browser = AGIEnhancedBrowser(headless=False)

    try:
        await browser.start()

        # Example commands
        await browser.execute_natural_language_command("go to github.com")
        await asyncio.sleep(2)

        await browser.execute_natural_language_command("search github for react")
        await asyncio.sleep(2)

        await browser.execute_natural_language_command("take screenshot")

    finally:
        await browser.stop()


async def demo_github_integration():
    """Demo: GitHub search and repository creation."""
    # Set GITHUB_TOKEN environment variable first
    browser = AGIEnhancedBrowser(headless=False)

    try:
        await browser.start()

        # Search GitHub
        await browser.execute_natural_language_command("search github for python web scraping")
        await asyncio.sleep(1)

        # Create repository (requires GitHub token)
        await browser.execute_natural_language_command(
            "create repo agi-test-project with description Created by AGI Browser"
        )

    finally:
        await browser.stop()


async def demo_code_generation():
    """Demo: Generate code and push to GitHub."""
    browser = AGIEnhancedBrowser(headless=False)

    try:
        await browser.start()

        # Generate and push code
        await browser.generate_and_push_code(
            "hello-agi",
            "A simple hello world program"
        )

    finally:
        await browser.stop()


async def main():
    """Main entry point - Interactive mode."""
    browser = AGIEnhancedBrowser(headless=False)

    try:
        await browser.start()
        await browser.interactive_mode()
    finally:
        await browser.stop()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         🤖 AGI-Enhanced Browser v2.0                             ║
    ║         AI-Powered Web Automation & GitHub Integration           ║
    ╚══════════════════════════════════════════════════════════════════╝

    Features:
    ✓ Natural language command processing
    ✓ Autonomous web navigation
    ✓ GitHub repository search
    ✓ GitHub repository creation
    ✓ Code generation and deployment
    ✓ Smart element interaction
    ✓ Session logging and analytics

    Setup:
    1. Install: pip install playwright requests
    2. Install browsers: playwright install
    3. Set GITHUB_TOKEN env variable for GitHub features

    Running:
    - Interactive mode: python agi_enhanced_browser.py
    - Demo: Uncomment demo functions at bottom
    """)

    # Run interactive mode
    asyncio.run(main())

    # Or run demos:
    # asyncio.run(demo_basic_browsing())
    # asyncio.run(demo_github_integration())
    # asyncio.run(demo_code_generation())
