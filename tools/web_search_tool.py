"""
Web Search Tool for Nexus AGI
Provides web search capabilities using multiple search engines
"""

from typing import Dict, Any, Optional, List
import os
from .base_tool import BaseTool, log_execution
from .http_fetch_tool import HTTPFetchTool
import logging

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Web search tool supporting multiple search engines

    Supported engines:
    - Google Custom Search Engine (requires API key)
    - SerpAPI (requires API key)
    - Bing Search API (requires API key)
    - DuckDuckGo (no API key required, limited)
    """

    def __init__(
        self,
        timeout: int = 15,
        max_retries: int = 2,
        rate_limit: Optional[int] = 50,
        **kwargs
    ):
        """
        Initialize Web Search Tool

        Configuration via environment variables:
        - GOOGLE_API_KEY: Google Custom Search API key
        - GOOGLE_CSE_ID: Google Custom Search Engine ID
        - SERPAPI_KEY: SerpAPI key
        - BING_SEARCH_API_KEY: Bing Search API key
        """
        super().__init__(timeout=timeout, max_retries=max_retries, rate_limit=rate_limit, **kwargs)

        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.bing_api_key = os.getenv('BING_SEARCH_API_KEY')

        self.http_client = HTTPFetchTool(timeout=timeout, max_retries=max_retries)

    @log_execution
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute web search

        Args:
            params: {
                'query': str (required) - Search query
                'engine': str (optional) - 'google', 'serpapi', 'bing', or 'duckduckgo' (default: auto)
                'max_results': int (optional) - Maximum results to return (default: 10)
                'language': str (optional) - Language code (default: 'en')
                'region': str (optional) - Region code (default: 'us')
            }

        Returns:
            {
                'success': bool,
                'query': str,
                'results': List[{
                    'title': str,
                    'url': str,
                    'snippet': str,
                    'position': int
                }],
                'total_results': int,
                'search_time': float,
                'engine': str
            }
        """
        self.validate_params(params)

        query = params['query']
        engine = params.get('engine', 'auto')
        max_results = params.get('max_results', 10)
        language = params.get('language', 'en')
        region = params.get('region', 'us')

        # Auto-select engine based on available API keys
        if engine == 'auto':
            engine = self._select_engine()

        # Execute search based on engine
        if engine == 'google':
            return self._search_google(query, max_results, language, region)
        elif engine == 'serpapi':
            return self._search_serpapi(query, max_results, language, region)
        elif engine == 'bing':
            return self._search_bing(query, max_results, language, region)
        elif engine == 'duckduckgo':
            return self._search_duckduckgo(query, max_results)
        else:
            return {
                'success': False,
                'error': f'Unsupported search engine: {engine}'
            }

    def _select_engine(self) -> str:
        """Auto-select best available search engine"""
        if self.google_api_key and self.google_cse_id:
            return 'google'
        elif self.serpapi_key:
            return 'serpapi'
        elif self.bing_api_key:
            return 'bing'
        else:
            logger.warning("No API keys configured, falling back to DuckDuckGo (limited results)")
            return 'duckduckgo'

    def _search_google(self, query: str, max_results: int, language: str, region: str) -> Dict[str, Any]:
        """Search using Google Custom Search Engine"""
        if not self.google_api_key or not self.google_cse_id:
            return {
                'success': False,
                'error': 'Google API key or CSE ID not configured'
            }

        url = "https://www.googleapis.com/customsearch/v1"
        params_dict = {
            'key': self.google_api_key,
            'cx': self.google_cse_id,
            'q': query,
            'num': min(max_results, 10),  # Google API max is 10 per request
            'hl': language,
            'gl': region
        }

        result = self.http_client.run({
            'url': url,
            'params': params_dict,
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        data = result.get('data', {})
        items = data.get('items', [])

        return {
            'success': True,
            'query': query,
            'results': [
                {
                    'title': item.get('title'),
                    'url': item.get('link'),
                    'snippet': item.get('snippet'),
                    'position': idx + 1
                }
                for idx, item in enumerate(items[:max_results])
            ],
            'total_results': int(data.get('searchInformation', {}).get('totalResults', 0)),
            'search_time': float(data.get('searchInformation', {}).get('searchTime', 0)),
            'engine': 'google'
        }

    def _search_serpapi(self, query: str, max_results: int, language: str, region: str) -> Dict[str, Any]:
        """Search using SerpAPI"""
        if not self.serpapi_key:
            return {
                'success': False,
                'error': 'SerpAPI key not configured'
            }

        url = "https://serpapi.com/search"
        params_dict = {
            'api_key': self.serpapi_key,
            'q': query,
            'num': max_results,
            'hl': language,
            'gl': region,
            'engine': 'google'
        }

        result = self.http_client.run({
            'url': url,
            'params': params_dict,
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        data = result.get('data', {})
        organic_results = data.get('organic_results', [])

        return {
            'success': True,
            'query': query,
            'results': [
                {
                    'title': item.get('title'),
                    'url': item.get('link'),
                    'snippet': item.get('snippet'),
                    'position': item.get('position', idx + 1)
                }
                for idx, item in enumerate(organic_results[:max_results])
            ],
            'total_results': len(organic_results),
            'search_time': data.get('search_metadata', {}).get('total_time_taken', 0),
            'engine': 'serpapi'
        }

    def _search_bing(self, query: str, max_results: int, language: str, region: str) -> Dict[str, Any]:
        """Search using Bing Search API"""
        if not self.bing_api_key:
            return {
                'success': False,
                'error': 'Bing API key not configured'
            }

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            'Ocp-Apim-Subscription-Key': self.bing_api_key
        }
        params_dict = {
            'q': query,
            'count': max_results,
            'mkt': f'{language}-{region}'
        }

        result = self.http_client.run({
            'url': url,
            'headers': headers,
            'params': params_dict,
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        data = result.get('data', {})
        web_pages = data.get('webPages', {})
        values = web_pages.get('value', [])

        return {
            'success': True,
            'query': query,
            'results': [
                {
                    'title': item.get('name'),
                    'url': item.get('url'),
                    'snippet': item.get('snippet'),
                    'position': idx + 1
                }
                for idx, item in enumerate(values[:max_results])
            ],
            'total_results': web_pages.get('totalEstimatedMatches', 0),
            'search_time': 0,  # Bing doesn't provide this
            'engine': 'bing'
        }

    def _search_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        """
        Search using DuckDuckGo (no API key required)

        Note: DuckDuckGo's instant answer API is limited and may not return web search results.
        This is a fallback option when no other API keys are available.
        """
        url = "https://api.duckduckgo.com/"
        params_dict = {
            'q': query,
            'format': 'json',
            'no_html': 1,
            'skip_disambig': 1
        }

        result = self.http_client.run({
            'url': url,
            'params': params_dict,
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        data = result.get('data', {})
        related_topics = data.get('RelatedTopics', [])

        # Extract results from related topics
        results = []
        for idx, topic in enumerate(related_topics[:max_results]):
            if isinstance(topic, dict) and 'FirstURL' in topic:
                results.append({
                    'title': topic.get('Text', '').split(' - ')[0] if topic.get('Text') else '',
                    'url': topic.get('FirstURL'),
                    'snippet': topic.get('Text', ''),
                    'position': idx + 1
                })

        return {
            'success': True,
            'query': query,
            'results': results,
            'total_results': len(results),
            'search_time': 0,
            'engine': 'duckduckgo',
            'warning': 'DuckDuckGo instant answers API has limited results. Consider using a proper search API for better results.'
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate search parameters"""
        if 'query' not in params:
            raise ValueError("Parameter 'query' is required")

        query = params['query']
        if not isinstance(query, str) or len(query.strip()) == 0:
            raise ValueError("Parameter 'query' must be a non-empty string")

        max_results = params.get('max_results', 10)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 100:
            raise ValueError("Parameter 'max_results' must be an integer between 1 and 100")

        engine = params.get('engine', 'auto')
        valid_engines = ['auto', 'google', 'serpapi', 'bing', 'duckduckgo']
        if engine not in valid_engines:
            raise ValueError(f"Parameter 'engine' must be one of {valid_engines}")

        return True

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for search parameters"""
        return {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query string',
                    'minLength': 1
                },
                'engine': {
                    'type': 'string',
                    'description': 'Search engine to use',
                    'enum': ['auto', 'google', 'serpapi', 'bing', 'duckduckgo'],
                    'default': 'auto'
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results to return',
                    'minimum': 1,
                    'maximum': 100,
                    'default': 10
                },
                'language': {
                    'type': 'string',
                    'description': 'Language code (e.g., en, es, fr)',
                    'default': 'en'
                },
                'region': {
                    'type': 'string',
                    'description': 'Region code (e.g., us, uk, ca)',
                    'default': 'us'
                }
            },
            'required': ['query']
        }

    def get_default_timeout(self) -> int:
        """Default timeout for search requests"""
        return 15


if __name__ == "__main__":
    # Test the Web Search Tool
    tool = WebSearchTool()

    print("=== Test: Web Search ===")
    result = tool.run({
        'query': 'artificial general intelligence',
        'max_results': 5,
        'engine': 'auto'
    })

    print(f"Success: {result.get('success')}")
    print(f"Engine: {result.get('engine')}")
    print(f"Total results: {result.get('total_results')}")

    if result.get('success'):
        print(f"\nTop {len(result.get('results', []))} results:")
        for res in result.get('results', []):
            print(f"\n{res['position']}. {res['title']}")
            print(f"   {res['url']}")
            print(f"   {res['snippet'][:100]}...")
    else:
        print(f"Error: {result.get('error')}")

    # Print usage stats
    print("\n=== Usage Stats ===")
    print(tool.get_usage_stats())
