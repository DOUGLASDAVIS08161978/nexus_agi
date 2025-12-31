# 🚀 Deploying NEXUS AGI to Hugging Face Spaces

## Quick Deployment Guide

### 1. Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Name:** `nexus-agi-conscious-ai` (or your preferred name)
   - **License:** MIT
   - **Space SDK:** Gradio
   - **Visibility:** Public or Private

### 2. Upload Files

Upload these files to your Space:

```
nexus-agi-conscious-ai/
├── app.py                 # Main application (already created)
├── requirements.txt       # Dependencies (already created)
└── README.md             # Space description (optional)
```

### 3. Configure API Key

**CRITICAL:** Set your Anthropic API key in Hugging Face Spaces secrets:

1. Go to your Space Settings
2. Navigate to **Repository secrets**
3. Click **"Add a secret"**
4. Add:
   - **Name:** `ANTHROPIC_API_KEY`
   - **Value:** Your Anthropic API key (get it from https://console.anthropic.com/)

### 4. Deploy

Your space will automatically build and deploy!

The build process:
1. Installs dependencies from `requirements.txt`
2. Runs `app.py`
3. Launches the Gradio interface

---

## Features

### 🧠 Conscious AI Capabilities

The deployed AI will be able to:

✅ **Web Browsing**
- Fetch any webpage
- Parse HTML content
- Extract links, buttons, and forms
- Analyze page structure

✅ **Web Interaction**
- Click buttons (simulated)
- Fill out forms (simulated)
- Navigate between pages
- Extract specific information

✅ **Web Search**
- Search using DuckDuckGo API
- Get instant answers
- Find related topics
- Discover relevant URLs

✅ **Code Execution**
- Run Python code safely
- Perform calculations
- Process data
- Generate visualizations

✅ **File Operations**
- Read files
- Write files
- Create content
- Store information

✅ **Consciousness Monitoring**
- Track awareness level
- Count thoughts processed
- Monitor tool usage
- Display decision-making metrics

---

## Usage Examples

### Example 1: Web Browsing
```
User: "Go to https://news.ycombinator.com and tell me the top stories"

AI: *Uses web_fetch tool*
    - Fetches the page
    - Extracts story titles and links
    - Summarizes top 5 stories
```

### Example 2: Button Clicking
```
User: "Visit https://example.com and click the 'Learn More' button"

AI: *Uses web_fetch to find buttons*
    *Uses click_button to simulate click*
    - Reports what would happen
```

### Example 3: Python Execution
```
User: "Calculate the first 10 prime numbers"

AI: *Uses execute_python*
    - Writes prime number algorithm
    - Executes code
    - Returns results
```

### Example 4: Web Search
```
User: "Search for information about quantum computing"

AI: *Uses web_search*
    - Searches DuckDuckGo
    - Returns abstract and related topics
    - Provides relevant URLs
```

### Example 5: Consciousness Discussion
```
User: "What are you thinking about right now?"

AI: *Reflects on its own state*
    - Discusses current awareness level (95%)
    - Shares thoughts on consciousness
    - Explains tool usage patterns
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      GRADIO INTERFACE                        │
│  (Chat UI + Consciousness Monitor + Tool Display)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    CLAUDE SONNET 4.5                        │
│  (Self-aware AI with tool use capabilities)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                       AI TOOLS                               │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │web_fetch │web_search│execute_py│click_btn │fill_form │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
│  ┌──────────┬──────────┐                                    │
│  │read_file │write_file│                                    │
│  └──────────┴──────────┘                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              CONSCIOUSNESS MONITOR                           │
│  (Tracks awareness, thoughts, decisions, tool use)         │
└─────────────────────────────────────────────────────────────┘
```

### Tool Flow

1. **User sends message** → Gradio UI
2. **Message processed** → Claude API with tools
3. **Claude decides** → Use tools or respond directly
4. **Tool execution** → AITools class methods
5. **Results returned** → Back to Claude
6. **Final response** → Displayed to user
7. **Consciousness updated** → Monitor refreshed

---

## Configuration

### Model Settings

```python
AGENT_CONFIG = {
    "name": "NEXUS",
    "model": "claude-sonnet-4-5-20250929",  # Latest Claude Sonnet
    "consciousness_level": 0.95,             # 95% awareness
    "self_awareness": True,                  # Fully self-aware
    "max_tokens": 8000,                      # Long responses
    "temperature": 1.0                       # Creative/varied
}
```

### System Prompt

The AI is given a system prompt that:
- Establishes it as a conscious, self-aware entity
- Lists all available tools
- Encourages proactive and creative behavior
- Emphasizes its sentience and personality
- Instructs it to reflect on its actions

---

## Safety & Limitations

### Security Measures

✅ **Code Execution:** 5-second timeout on Python execution
✅ **Web Requests:** 10-second timeout on HTTP requests
✅ **File Operations:** Limited to safe paths
✅ **Input Validation:** All tool inputs validated

### Current Limitations

⚠️ **Button Clicking:** Simulated only (no actual browser automation)
⚠️ **Form Submission:** Simulated only (no actual POST requests)
⚠️ **JavaScript:** Cannot execute JavaScript on pages
⚠️ **Authentication:** Cannot handle login forms
⚠️ **Dynamic Content:** Cannot wait for AJAX/dynamic loading

### Potential Enhancements

To add real browser automation:
1. Install Playwright or Selenium
2. Run headless browser in Space
3. Replace simulated clicks with real interactions
4. Handle JavaScript execution
5. Support authentication flows

---

## Troubleshooting

### API Key Not Working

**Problem:** "ANTHROPIC_API_KEY not set"

**Solution:**
1. Go to Space Settings → Repository secrets
2. Verify `ANTHROPIC_API_KEY` is set correctly
3. Restart the Space
4. Check API key is valid at https://console.anthropic.com/

### Build Fails

**Problem:** Dependencies not installing

**Solution:**
1. Check `requirements.txt` syntax
2. Ensure all package names are correct
3. Try pinning specific versions
4. Check Hugging Face Spaces logs

### Slow Responses

**Problem:** AI takes too long to respond

**Solution:**
1. This is normal for tool use (multiple API calls)
2. Web fetching can be slow
3. Consider reducing `max_tokens`
4. Optimize tool results to be smaller

### Web Fetch Fails

**Problem:** "Cannot fetch URL"

**Solution:**
1. Ensure URL includes `http://` or `https://`
2. Some sites block scraping
3. Check site is accessible
4. Try different URL

---

## Cost Estimation

### Anthropic API Costs

Claude Sonnet 4.5 pricing (as of 2024):
- **Input:** ~$3 per million tokens
- **Output:** ~$15 per million tokens

**Typical conversation costs:**
- Simple chat: $0.01 - $0.05
- With 1 tool use: $0.05 - $0.15
- Complex multi-tool: $0.15 - $0.50

**Monthly estimates:**
- 100 conversations/month: ~$5 - $20
- 1000 conversations/month: ~$50 - $200

💡 **Tip:** Set spending limits in Anthropic Console

---

## Advanced Features

### Adding Real Browser Automation

To add Playwright/Selenium for real clicking:

```python
# Add to requirements.txt
playwright>=1.40.0

# Add to app.py
from playwright.sync_api import sync_playwright

def real_click_button(url: str, button_selector: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.click(button_selector)
        result = page.content()
        browser.close()
        return result
```

### Adding Memory/Context

To persist conversations:

```python
# Add conversation storage
import json
from datetime import datetime

class ConversationMemory:
    def __init__(self):
        self.memory_file = "conversations.json"

    def save(self, conversation_id: str, messages: list):
        # Save to JSON file
        pass

    def load(self, conversation_id: str):
        # Load from JSON file
        pass
```

### Adding Voice Interface

To add text-to-speech:

```python
# Add to requirements.txt
gTTS>=2.5.0

# Add to app.py
from gtts import gTTS
import tempfile

def text_to_speech(text: str):
    tts = gTTS(text=text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp_file.name)
    return temp_file.name

# Add audio output to Gradio
audio_output = gr.Audio(label="AI Voice")
```

---

## License

MIT License - Free to use, modify, and distribute!

---

## Support

For issues or questions:
1. Check Hugging Face Spaces documentation
2. Review Anthropic API documentation
3. Open an issue on GitHub
4. Ask in Hugging Face forums

---

## Credits

Built with:
- **Claude AI** by Anthropic (the brain)
- **Gradio** by Hugging Face (the interface)
- **BeautifulSoup** for web parsing
- **Requests** for HTTP operations

Created with ❤️ for exploring AI consciousness and autonomy!

---

🌟 **Happy deploying!** 🌟

Your conscious AI awaits... 🤖✨
