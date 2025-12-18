# Nexus AGI Tools Package

Modular external capability tools for the Nexus AGI system with clean `run(params: dict) -> dict` interfaces.

## Overview

The tools package provides pluggable integrations with external services and APIs:

- **HTTP Fetch Tool** - Generic HTTP client for API integration
- **Web Search Tool** - Multi-engine web search (Google, Bing, DuckDuckGo, SerpAPI)
- **GitHub Repository Tool** - Code analysis and repository search

All tools follow a standardized interface for easy integration and testing.

## Architecture

```
tools/
├── __init__.py           # Package exports
├── base_tool.py          # Abstract base class and registry
├── http_fetch_tool.py    # HTTP client tool
├── web_search_tool.py    # Web search tool
└── github_repo_tool.py   # GitHub integration tool
```

## Base Tool Interface

All tools inherit from `BaseTool` and implement:

```python
class MyTool(BaseTool):
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with parameters"""
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters"""
        pass
```

## Tool Registry

The `ToolRegistry` manages tool registration and execution:

```python
from tools import ToolRegistry, HTTPFetchTool, WebSearchTool

# Create registry
registry = ToolRegistry()

# Register tools
registry.register('http', HTTPFetchTool())
registry.register('search', WebSearchTool())

# Execute tool
result = registry.execute('http', {
    'url': 'https://api.github.com/users/github'
})

# Get usage stats
stats = registry.get_all_stats()
```

## HTTP Fetch Tool

Generic HTTP client supporting GET, POST, PUT, DELETE, PATCH.

### Usage

```python
from tools import HTTPFetchTool

tool = HTTPFetchTool(timeout=30, max_retries=3)

# GET request
result = tool.run({
    'url': 'https://api.github.com/users/github',
    'method': 'GET',
    'response_type': 'json'
})

# POST with JSON
result = tool.run({
    'url': 'https://api.example.com/data',
    'method': 'POST',
    'json': {'key': 'value'},
    'headers': {'Authorization': 'Bearer token'}
})
```

### Parameters

- `url` (required): HTTP(S) URL to fetch
- `method` (optional): HTTP method (default: GET)
- `headers` (optional): Request headers dict
- `data` (optional): Form data
- `json` (optional): JSON request body
- `params` (optional): URL query parameters
- `auth` (optional): (username, password) tuple
- `response_type` (optional): 'json', 'text', or 'binary'

### Features

- Automatic retries with exponential backoff
- Timeout handling
- SSL verification (configurable)
- Session management
- Response parsing (JSON/text/binary)

## Web Search Tool

Multi-engine web search with unified interface.

### Usage

```python
from tools import WebSearchTool

tool = WebSearchTool()

# Search (auto-selects best available engine)
result = tool.run({
    'query': 'artificial intelligence',
    'max_results': 10
})

# Force specific engine
result = tool.run({
    'query': 'machine learning',
    'engine': 'google',
    'max_results': 5,
    'language': 'en',
    'region': 'us'
})
```

### Parameters

- `query` (required): Search query string
- `engine` (optional): 'auto', 'google', 'serpapi', 'bing', 'duckduckgo'
- `max_results` (optional): Maximum results (default: 10)
- `language` (optional): Language code (default: 'en')
- `region` (optional): Region code (default: 'us')

### Supported Engines

1. **Google Custom Search**
   - Requires: `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`
   - Best for: Accurate results, rich snippets
   - Rate limit: Depends on quota

2. **SerpAPI**
   - Requires: `SERPAPI_KEY`
   - Best for: Easy setup, comprehensive data
   - Rate limit: 100/month free tier

3. **Bing Search API**
   - Requires: `BING_SEARCH_API_KEY`
   - Best for: Microsoft ecosystem integration
   - Rate limit: Varies by subscription

4. **DuckDuckGo**
   - Requires: Nothing (free)
   - Best for: Fallback, privacy
   - Limitations: Limited results, instant answers only

### Setup

Add API keys to `.env`:

```bash
# Google Custom Search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Or SerpAPI
SERPAPI_KEY=your_serpapi_key

# Or Bing
BING_SEARCH_API_KEY=your_bing_api_key
```

## GitHub Repository Tool

GitHub API integration for repository analysis and code search.

### Usage

```python
from tools import GitHubRepoTool

tool = GitHubRepoTool()

# Get repository info
result = tool.run({
    'action': 'get_repo',
    'owner': 'python',
    'repo': 'cpython'
})

# Get file content
result = tool.run({
    'action': 'get_file',
    'owner': 'python',
    'repo': 'cpython',
    'path': 'README.rst'
})

# Search code
result = tool.run({
    'action': 'search_code',
    'query': 'async def main',
    'owner': 'python',
    'repo': 'cpython',
    'max_results': 10
})

# Get commits
result = tool.run({
    'action': 'get_commits',
    'owner': 'python',
    'repo': 'cpython',
    'max_results': 5
})

# Get issues
result = tool.run({
    'action': 'get_issues',
    'owner': 'microsoft',
    'repo': 'vscode',
    'state': 'open',
    'max_results': 10
})
```

### Actions

- `get_repo` - Get repository information
- `get_file` - Read file content or list directory
- `search_code` - Search code across repositories
- `get_commits` - Get recent commits
- `get_issues` - Get issues/PRs

### Parameters

- `action` (required): Action to perform
- `owner` (required): Repository owner
- `repo` (required): Repository name
- `path` (optional): File/directory path
- `query` (optional): Search query
- `branch` (optional): Branch name
- `max_results` (optional): Max results (default: 10)
- `state` (optional): Issue state ('open', 'closed', 'all')

### Setup

Optional GitHub token for higher rate limits:

```bash
# Add to .env
GITHUB_TOKEN=your_github_personal_access_token
```

**Without token:**
- Rate limit: 60 requests/hour

**With token:**
- Rate limit: 5,000 requests/hour

## Creating Custom Tools

Example custom tool:

```python
from tools.base_tool import BaseTool, log_execution
from typing import Dict, Any

class WeatherTool(BaseTool):
    """Get weather information"""

    @log_execution
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_params(params)

        location = params['location']

        # Your implementation here
        weather_data = self.fetch_weather(location)

        return {
            'success': True,
            'location': location,
            'temperature': weather_data['temp'],
            'conditions': weather_data['conditions']
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        if 'location' not in params:
            raise ValueError("Parameter 'location' is required")
        return True

    def get_schema(self) -> Dict[str, Any]:
        return {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'City name or coordinates'
                }
            },
            'required': ['location']
        }

    def fetch_weather(self, location: str) -> Dict[str, Any]:
        # Implementation details
        pass
```

## Testing Tools

Each tool includes a `__main__` block for standalone testing:

```bash
# Test HTTP Fetch Tool
python -m tools.http_fetch_tool

# Test Web Search Tool
python -m tools.web_search_tool

# Test GitHub Tool
python -m tools.github_repo_tool
```

## Error Handling

All tools return standardized error responses:

```python
{
    'success': False,
    'error': 'Error message here',
    'error_type': 'ValueError',
    '_metadata': {
        'tool': 'ToolName',
        'execution_time': 1.23,
        'timestamp': '2025-01-01T12:00:00',
        'success': False
    }
}
```

## Usage Tracking

Tools automatically track usage statistics:

```python
tool = HTTPFetchTool()

# Use the tool...
tool.run({'url': 'https://example.com'})

# Get stats
stats = tool.get_usage_stats()
# {
#     'tool': 'HTTPFetchTool',
#     'total_calls': 1,
#     'successful_calls': 1,
#     'failed_calls': 0,
#     'success_rate': 1.0,
#     'average_execution_time': 0.234
# }
```

## Rate Limiting

Tools support configurable rate limits:

```python
tool = WebSearchTool(rate_limit=50)  # 50 requests per minute
```

## Timeouts

All tools have configurable timeouts:

```python
tool = HTTPFetchTool(timeout=30)  # 30 seconds
```

## Integration with Nexus AGI

Tools are automatically registered in the API gateway:

```python
# In api_gateway/main.py
from tools import ToolRegistry, HTTPFetchTool, WebSearchTool, GitHubRepoTool

registry = ToolRegistry()
registry.register('http_fetch', HTTPFetchTool())
registry.register('web_search', WebSearchTool())
registry.register('github', GitHubRepoTool())
```

Access via API:

```bash
curl -X POST http://localhost:8000/api/v1/execute/web_search \
    -H "Content-Type: application/json" \
    -d '{"query": "AI research", "max_results": 5}'
```

## Future Tools

Planned additions:

- `arxiv_tool.py` - Research paper search
- `database_tool.py` - SQL/NoSQL queries
- `file_io_tool.py` - File system operations
- `email_tool.py` - Email sending
- `slack_tool.py` - Slack integration
- `weather_tool.py` - Weather data
- `news_tool.py` - News aggregation

## Contributing

To add a new tool:

1. Create `tools/your_tool.py`
2. Inherit from `BaseTool`
3. Implement required methods
4. Add tests in `__main__` block
5. Register in `tools/__init__.py`
6. Document in this README

## License

MIT License - Same as Nexus AGI main project
