# Web Browser Agent - AGI Browser Automation

Autonomous web browsing and interaction capabilities for the Nexus AGI system.

## Features

✅ **Web Navigation** - Navigate to any URL
✅ **Search Integration** - Google search automation
✅ **Element Interaction** - Click buttons, fill forms
✅ **Data Extraction** - Extract text from page elements
✅ **Screenshots** - Capture page screenshots
✅ **Human-Directed** - Accept commands from humans
✅ **Workflow Automation** - Execute multi-step workflows

## Installation

```bash
# Install Playwright
pip install playwright

# Install browser binaries
playwright install chromium
```

## Quick Start

### 1. Basic Usage

```python
import asyncio
from web_browser_agent import WebBrowserAgent

async def main():
    agent = WebBrowserAgent(headless=True)

    await agent.start()
    await agent.navigate("https://example.com")
    await agent.screenshot("page.png")
    await agent.stop()

asyncio.run(main())
```

### 2. Search and Click

```python
async def search_example():
    agent = WebBrowserAgent(headless=False)  # Show browser

    await agent.start()
    await agent.search_google("AI research")
    await asyncio.sleep(2)
    await agent.click_element("h3")  # Click first result
    await agent.stop()

asyncio.run(search_example())
```

### 3. Workflow Automation

```python
workflow = [
    {'action': 'navigate', 'url': 'https://example.com'},
    {'action': 'click_button', 'text': 'Learn More'},
    {'action': 'wait', 'seconds': 2},
    {'action': 'screenshot', 'path': 'result.png'}
]

async def run_workflow():
    agent = WebBrowserAgent()
    await agent.start()
    results = await agent.execute_workflow(workflow)
    await agent.stop()

asyncio.run(run_workflow())
```

## Available Actions

### Navigation
```python
{'action': 'navigate', 'url': 'https://example.com'}
```

### Search
```python
{'action': 'search', 'query': 'search term'}
```

### Click Element
```python
{'action': 'click', 'selector': 'button.submit'}
```

### Click Button by Text
```python
{'action': 'click_button', 'text': 'Submit'}
```

### Fill Form Field
```python
{'action': 'fill', 'selector': 'input[name="email"]', 'value': 'test@example.com'}
```

### Extract Text
```python
{'action': 'get_text', 'selector': 'h1'}
```

### Screenshot
```python
{'action': 'screenshot', 'path': 'output.png'}
```

### Wait
```python
{'action': 'wait', 'seconds': 3}
```

## Running Examples

```bash
# Run interactive menu
python browser_automation_examples.py

# Or run specific examples:
python -c "import asyncio; from browser_automation_examples import example_1_simple_navigation; asyncio.run(example_1_simple_navigation())"
```

## Human-Directed Mode

Run the agent in interactive mode where humans can give commands:

```bash
python browser_automation_examples.py
# Choose option 6

# Then use commands like:
> navigate https://example.com
> search artificial intelligence
> click_button Sign In
> screenshot
> quit
```

## Workflow JSON Files

Create workflow JSON files for repeatable automation:

**workflow.json:**
```json
[
  {
    "action": "navigate",
    "url": "https://example.com"
  },
  {
    "action": "click_button",
    "text": "Get Started"
  },
  {
    "action": "wait",
    "seconds": 2
  },
  {
    "action": "screenshot",
    "path": "completed.png"
  }
]
```

**Run the workflow:**
```python
from browser_automation_examples import run_workflow_from_file
import asyncio

asyncio.run(run_workflow_from_file('workflow.json'))
```

## API Reference

### WebBrowserAgent Class

```python
agent = WebBrowserAgent(headless=True)
```

**Methods:**
- `start()` - Start browser instance
- `stop()` - Stop browser instance
- `navigate(url)` - Navigate to URL
- `search_google(query)` - Search Google
- `click_element(selector)` - Click element by CSS selector
- `click_button_by_text(text)` - Click button by text
- `fill_form_field(selector, value)` - Fill input field
- `get_text(selector)` - Extract text from element
- `screenshot(path)` - Take screenshot
- `execute_command(command)` - Execute single command
- `execute_workflow(workflow)` - Execute workflow
- `save_log(path)` - Save interaction log

## Use Cases

### 1. Web Scraping
Extract data from multiple pages automatically.

### 2. Testing
Automate website testing and regression checks.

### 3. Research
Gather information from multiple sources.

### 4. Data Collection
Collect data from web forms and pages.

### 5. Monitoring
Monitor websites for changes.

### 6. Task Automation
Automate repetitive web tasks.

## Important Notes

⚠️ **Ethical Use:**
- Respect websites' Terms of Service
- Don't overload servers with requests
- Check robots.txt before scraping
- Use appropriate delays between requests

⚠️ **Rate Limiting:**
- Add wait times between actions
- Respect website rate limits
- Use headless mode for efficiency

⚠️ **Authentication:**
- Don't store credentials in code
- Use environment variables
- Be careful with sensitive data

## Troubleshooting

### Browser Not Found
```bash
playwright install chromium
```

### Selector Not Found
- Check the CSS selector is correct
- Add wait time before clicking
- Use browser DevTools to find selectors

### Connection Timeout
- Increase timeout values
- Check internet connection
- Verify URL is correct

## Examples

See `browser_automation_examples.py` for:
1. Simple navigation
2. Form interaction
3. Data extraction
4. Button clicking
5. Custom workflows
6. Human-directed mode

## Integration with Nexus AGI

The browser agent can be integrated with the main AGI system:

```python
from web_browser_agent import WebBrowserAgent
from nexus_agi import MetaAlgorithm_NexusCore

# Use AGI to generate web automation strategies
# Use browser agent to execute them
```

## License

MIT License - See LICENSE file

## Support

For issues or questions, see the main Nexus AGI repository.
