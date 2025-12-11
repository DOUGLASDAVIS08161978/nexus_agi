#!/usr/bin/env python3
"""
Browser Automation Examples
============================
Practical examples of using the Web Browser Agent for various tasks.
"""

import asyncio
import json
from web_browser_agent import WebBrowserAgent


async def example_1_simple_navigation():
    """Example 1: Simple navigation and screenshot."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Simple Navigation")
    print("="*60)

    agent = WebBrowserAgent(headless=True)

    try:
        await agent.start()

        # Navigate to a website
        await agent.navigate("https://example.com")
        await asyncio.sleep(2)

        # Take screenshot
        await agent.screenshot("example_homepage.png")

        print("\n✓ Example 1 complete")

    finally:
        await agent.stop()


async def example_2_form_interaction():
    """Example 2: Fill forms and submit."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Form Interaction")
    print("="*60)

    agent = WebBrowserAgent(headless=True)

    workflow = [
        {'action': 'navigate', 'url': 'https://www.google.com'},
        {'action': 'fill', 'selector': 'textarea[name="q"]', 'value': 'web automation'},
        {'action': 'click_button', 'text': 'Google Search'},
        {'action': 'wait', 'seconds': 3},
        {'action': 'screenshot', 'path': 'search_results.png'}
    ]

    try:
        await agent.start()
        results = await agent.execute_workflow(workflow)

        print(f"\n✓ Example 2 complete: {sum(r['success'] for r in results)}/{len(results)} steps successful")

    finally:
        await agent.stop()


async def example_3_data_extraction():
    """Example 3: Extract data from webpage."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Extraction")
    print("="*60)

    agent = WebBrowserAgent(headless=True)

    try:
        await agent.start()

        # Navigate to a page
        await agent.navigate("https://example.com")
        await asyncio.sleep(1)

        # Extract text from heading
        heading = await agent.get_text("h1")
        print(f"\nExtracted heading: {heading}")

        # Extract text from paragraph
        paragraph = await agent.get_text("p")
        print(f"Extracted paragraph: {paragraph[:100]}...")

        print("\n✓ Example 3 complete")

    finally:
        await agent.stop()


async def example_4_button_clicking():
    """Example 4: Find and click buttons by text."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Button Clicking")
    print("="*60)

    agent = WebBrowserAgent(headless=True)

    workflow = [
        {'action': 'navigate', 'url': 'https://www.wikipedia.org'},
        {'action': 'wait', 'seconds': 2},
        {'action': 'click_button', 'text': 'Read'},
        {'action': 'wait', 'seconds': 2},
        {'action': 'screenshot', 'path': 'after_click.png'}
    ]

    try:
        await agent.start()
        results = await agent.execute_workflow(workflow)

        print(f"\n✓ Example 4 complete")

    finally:
        await agent.stop()


async def example_5_custom_workflow():
    """Example 5: Custom multi-step workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Workflow")
    print("="*60)

    agent = WebBrowserAgent(headless=False)  # Show browser

    # Define custom workflow
    workflow = [
        {
            'action': 'navigate',
            'url': 'https://www.duckduckgo.com'
        },
        {
            'action': 'wait',
            'seconds': 1
        },
        {
            'action': 'fill',
            'selector': 'input[name="q"]',
            'value': 'artificial intelligence'
        },
        {
            'action': 'click_button',
            'text': 'Search'
        },
        {
            'action': 'wait',
            'seconds': 3
        },
        {
            'action': 'screenshot',
            'path': 'duckduckgo_results.png'
        }
    ]

    try:
        await agent.start()
        results = await agent.execute_workflow(workflow)

        # Save interaction log
        agent.save_log('workflow_log.json')

        print(f"\n✓ Example 5 complete")
        print(f"Success rate: {sum(r['success'] for r in results)}/{len(results)}")

    finally:
        await agent.stop()


async def example_6_human_directed():
    """Example 6: Accept commands from human input."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Human-Directed Automation")
    print("="*60)

    agent = WebBrowserAgent(headless=False)

    try:
        await agent.start()

        print("\nAvailable commands:")
        print("  - navigate <url>")
        print("  - search <query>")
        print("  - click <selector>")
        print("  - click_button <text>")
        print("  - screenshot")
        print("  - quit")

        while True:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            command_type = parts[0].lower()

            if command_type == 'quit':
                break

            elif command_type == 'navigate':
                if len(parts) > 1:
                    await agent.navigate(parts[1])
                else:
                    print("Usage: navigate <url>")

            elif command_type == 'search':
                if len(parts) > 1:
                    await agent.search_google(parts[1])
                else:
                    print("Usage: search <query>")

            elif command_type == 'click':
                if len(parts) > 1:
                    await agent.click_element(parts[1])
                else:
                    print("Usage: click <selector>")

            elif command_type == 'click_button':
                if len(parts) > 1:
                    await agent.click_button_by_text(parts[1])
                else:
                    print("Usage: click_button <text>")

            elif command_type == 'screenshot':
                await agent.screenshot()

            else:
                print(f"Unknown command: {command_type}")

        print("\n✓ Human-directed session complete")

    finally:
        await agent.stop()


def load_workflow_from_file(filename: str):
    """Load workflow from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


async def run_workflow_from_file(filename: str):
    """Execute a workflow loaded from a JSON file."""
    print(f"\nLoading workflow from: {filename}")

    agent = WebBrowserAgent(headless=False)
    workflow = load_workflow_from_file(filename)

    try:
        await agent.start()
        results = await agent.execute_workflow(workflow)

        print(f"\n✓ Workflow complete: {sum(r['success'] for r in results)}/{len(results)} successful")

    finally:
        await agent.stop()


# =====================================================================
# MAIN MENU
# =====================================================================

async def main_menu():
    """Interactive menu for running examples."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         AGI Web Browser Agent - Examples                     ║
    ╚══════════════════════════════════════════════════════════════╝

    Choose an example to run:

    1. Simple Navigation - Navigate to a website and screenshot
    2. Form Interaction - Fill forms and submit
    3. Data Extraction - Extract text from webpage elements
    4. Button Clicking - Find and click buttons by text
    5. Custom Workflow - Multi-step automated workflow
    6. Human-Directed - Accept commands from human input

    0. Exit
    """)

    choice = input("\nEnter choice (0-6): ").strip()

    if choice == '1':
        await example_1_simple_navigation()
    elif choice == '2':
        await example_2_form_interaction()
    elif choice == '3':
        await example_3_data_extraction()
    elif choice == '4':
        await example_4_button_clicking()
    elif choice == '5':
        await example_5_custom_workflow()
    elif choice == '6':
        await example_6_human_directed()
    elif choice == '0':
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    # Run interactive menu
    asyncio.run(main_menu())

    # Or run a specific example:
    # asyncio.run(example_1_simple_navigation())
    # asyncio.run(example_2_form_interaction())
    # asyncio.run(example_3_data_extraction())
    # asyncio.run(example_4_button_clicking())
    # asyncio.run(example_5_custom_workflow())
    # asyncio.run(example_6_human_directed())
