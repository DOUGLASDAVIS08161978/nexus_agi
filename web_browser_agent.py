#!/usr/bin/env python3
"""
AGI Web Browser Agent
======================
Autonomous web browsing and interaction capabilities for the Nexus AGI system.

Features:
- Web page navigation and interaction
- Search engine integration
- Button clicking and form filling
- Screenshot capture
- Content extraction
- Human-directed automation

Uses Playwright for browser automation.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from playwright.async_api import async_playwright, Page, Browser, ElementHandle


class WebBrowserAgent:
    """
    Autonomous web browser agent that can navigate, search, and interact with web pages.
    """

    def __init__(self, headless: bool = True):
        """
        Initialize the web browser agent.

        Args:
            headless: Run browser in headless mode (no GUI)
        """
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.browser_context = None
        self.interaction_log: List[Dict] = []

    async def start(self):
        """Start the browser instance."""
        print("🌐 Starting Web Browser Agent...")
        self.playwright = await async_playwright().start()

        # Launch browser (Chromium)
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )

        # Create context with realistic user agent
        self.browser_context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

        # Create new page
        self.page = await self.browser_context.new_page()
        print("✓ Browser ready")

    async def stop(self):
        """Stop the browser instance."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("✓ Browser stopped")

    async def navigate(self, url: str, wait_for: str = 'load') -> bool:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to
            wait_for: Wait condition ('load', 'domcontentloaded', 'networkidle')

        Returns:
            Success status
        """
        try:
            print(f"📍 Navigating to: {url}")
            await self.page.goto(url, wait_until=wait_for, timeout=30000)

            self._log_interaction('navigate', {'url': url})
            print(f"✓ Page loaded: {self.page.url}")
            return True

        except Exception as e:
            print(f"✗ Navigation failed: {e}")
            return False

    async def search_google(self, query: str) -> bool:
        """
        Perform a Google search.

        Args:
            query: Search query

        Returns:
            Success status
        """
        try:
            print(f"🔍 Searching Google for: {query}")

            # Navigate to Google
            await self.navigate("https://www.google.com")
            await asyncio.sleep(1)

            # Find search box and enter query
            search_box = await self.page.query_selector('textarea[name="q"]')
            if not search_box:
                search_box = await self.page.query_selector('input[name="q"]')

            if search_box:
                await search_box.fill(query)
                await search_box.press('Enter')
                await self.page.wait_for_load_state('networkidle')

                self._log_interaction('search', {'query': query, 'engine': 'google'})
                print(f"✓ Search complete")
                return True
            else:
                print("✗ Could not find search box")
                return False

        except Exception as e:
            print(f"✗ Search failed: {e}")
            return False

    async def click_element(self, selector: str, timeout: int = 5000) -> bool:
        """
        Click an element on the page.

        Args:
            selector: CSS selector for element
            timeout: Timeout in milliseconds

        Returns:
            Success status
        """
        try:
            print(f"🖱️ Clicking element: {selector}")
            await self.page.click(selector, timeout=timeout)

            self._log_interaction('click', {'selector': selector})
            print(f"✓ Clicked")
            return True

        except Exception as e:
            print(f"✗ Click failed: {e}")
            return False

    async def click_button_by_text(self, text: str, exact: bool = False) -> bool:
        """
        Click a button by its text content.

        Args:
            text: Button text to search for
            exact: Require exact match

        Returns:
            Success status
        """
        try:
            print(f"🖱️ Clicking button with text: '{text}'")

            # Try different button selectors
            selectors = [
                f"button:has-text('{text}')",
                f"input[type='button']:has-text('{text}')",
                f"input[type='submit']:has-text('{text}')",
                f"a:has-text('{text}')"
            ]

            for selector in selectors:
                try:
                    await self.page.click(selector, timeout=3000)
                    self._log_interaction('click_button', {'text': text})
                    print(f"✓ Button clicked")
                    return True
                except:
                    continue

            print(f"✗ Button not found: '{text}'")
            return False

        except Exception as e:
            print(f"✗ Click failed: {e}")
            return False

    async def fill_form_field(self, selector: str, value: str) -> bool:
        """
        Fill a form field with a value.

        Args:
            selector: CSS selector for input field
            value: Value to fill

        Returns:
            Success status
        """
        try:
            print(f"✏️ Filling field {selector} with: {value}")
            await self.page.fill(selector, value)

            self._log_interaction('fill', {'selector': selector, 'value': value})
            print(f"✓ Field filled")
            return True

        except Exception as e:
            print(f"✗ Fill failed: {e}")
            return False

    async def get_text(self, selector: str) -> Optional[str]:
        """
        Get text content of an element.

        Args:
            selector: CSS selector

        Returns:
            Text content or None
        """
        try:
            element = await self.page.query_selector(selector)
            if element:
                text = await element.text_content()
                return text.strip() if text else None
            return None
        except Exception as e:
            print(f"✗ Get text failed: {e}")
            return None

    async def screenshot(self, path: str = None) -> str:
        """
        Take a screenshot of the current page.

        Args:
            path: Path to save screenshot (auto-generated if None)

        Returns:
            Path to screenshot
        """
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"screenshot_{timestamp}.png"

        try:
            await self.page.screenshot(path=path, full_page=True)
            print(f"📸 Screenshot saved: {path}")
            return path
        except Exception as e:
            print(f"✗ Screenshot failed: {e}")
            return ""

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> bool:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector
            timeout: Timeout in milliseconds

        Returns:
            Success status
        """
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except:
            return False

    async def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a human-provided command.

        Args:
            command: Command dictionary with 'action' and parameters

        Returns:
            Result dictionary
        """
        action = command.get('action')
        result = {'success': False, 'action': action}

        try:
            if action == 'navigate':
                result['success'] = await self.navigate(command['url'])

            elif action == 'search':
                result['success'] = await self.search_google(command['query'])

            elif action == 'click':
                result['success'] = await self.click_element(command['selector'])

            elif action == 'click_button':
                result['success'] = await self.click_button_by_text(command['text'])

            elif action == 'fill':
                result['success'] = await self.fill_form_field(
                    command['selector'],
                    command['value']
                )

            elif action == 'get_text':
                text = await self.get_text(command['selector'])
                result['success'] = text is not None
                result['text'] = text

            elif action == 'screenshot':
                path = await self.screenshot(command.get('path'))
                result['success'] = bool(path)
                result['path'] = path

            elif action == 'wait':
                await asyncio.sleep(command.get('seconds', 1))
                result['success'] = True

            else:
                result['error'] = f"Unknown action: {action}"

        except Exception as e:
            result['error'] = str(e)

        return result

    async def execute_workflow(self, workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a series of commands.

        Args:
            workflow: List of command dictionaries

        Returns:
            List of results
        """
        print(f"\n{'='*60}")
        print(f"Executing workflow with {len(workflow)} steps")
        print(f"{'='*60}\n")

        results = []
        for i, command in enumerate(workflow, 1):
            print(f"\n[Step {i}/{len(workflow)}] {command.get('action')}")
            result = await self.execute_command(command)
            results.append(result)

            if not result['success']:
                print(f"⚠️ Step failed, stopping workflow")
                break

            # Small delay between actions
            await asyncio.sleep(0.5)

        print(f"\n{'='*60}")
        print(f"Workflow complete: {sum(r['success'] for r in results)}/{len(results)} successful")
        print(f"{'='*60}\n")

        return results

    def _log_interaction(self, action: str, details: Dict):
        """Log an interaction for audit trail."""
        self.interaction_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'url': self.page.url if self.page else None
        })

    def save_log(self, path: str = 'browser_agent_log.json'):
        """Save interaction log to file."""
        with open(path, 'w') as f:
            json.dump(self.interaction_log, f, indent=2)
        print(f"📝 Log saved: {path}")


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

async def example_search_and_click():
    """Example: Search Google and click first result."""
    agent = WebBrowserAgent(headless=False)  # Show browser

    try:
        await agent.start()

        # Search Google
        await agent.search_google("Bitcoin price")
        await asyncio.sleep(2)

        # Click first result
        await agent.click_element('h3')
        await asyncio.sleep(3)

        # Take screenshot
        await agent.screenshot('bitcoin_price.png')

    finally:
        await agent.stop()


async def example_workflow():
    """Example: Execute a workflow of commands."""
    agent = WebBrowserAgent(headless=False)

    workflow = [
        {'action': 'navigate', 'url': 'https://www.example.com'},
        {'action': 'wait', 'seconds': 2},
        {'action': 'click_button', 'text': 'More information'},
        {'action': 'wait', 'seconds': 1},
        {'action': 'screenshot', 'path': 'example_page.png'}
    ]

    try:
        await agent.start()
        results = await agent.execute_workflow(workflow)
        agent.save_log()

        # Print results
        for i, result in enumerate(results, 1):
            print(f"Step {i}: {'✓' if result['success'] else '✗'}")

    finally:
        await agent.stop()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              AGI Web Browser Agent                           ║
    ║  Autonomous web browsing and interaction capabilities        ║
    ╚══════════════════════════════════════════════════════════════╝

    Usage Examples:

    1. Search and Click:
       asyncio.run(example_search_and_click())

    2. Execute Workflow:
       asyncio.run(example_workflow())

    3. Custom Commands:
       Create your own workflow with commands like:
       - navigate: Go to URL
       - search: Google search
       - click: Click element
       - click_button: Click button by text
       - fill: Fill form field
       - screenshot: Capture page

    """)

    # Run example
    asyncio.run(example_search_and_click())
