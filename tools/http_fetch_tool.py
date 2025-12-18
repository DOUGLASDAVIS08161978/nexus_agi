"""
HTTP Fetch Tool for Nexus AGI
Generic HTTP client for API integration and web content fetching
"""

import requests
from typing import Dict, Any, Optional
import time
from .base_tool import BaseTool, log_execution
import logging

logger = logging.getLogger(__name__)


class HTTPFetchTool(BaseTool):
    """
    Generic HTTP client tool for fetching data from APIs and websites

    Supports:
    - GET, POST, PUT, DELETE, PATCH methods
    - Custom headers and authentication
    - Automatic retries with exponential backoff
    - Timeout handling
    - Response parsing (JSON, text, binary)
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit: Optional[int] = None,
        verify_ssl: bool = True,
        **kwargs
    ):
        """
        Initialize HTTP Fetch Tool

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit: Maximum requests per minute
            verify_ssl: Whether to verify SSL certificates
        """
        super().__init__(timeout=timeout, max_retries=max_retries, rate_limit=rate_limit, **kwargs)
        self.verify_ssl = verify_ssl
        self.session = requests.Session()

    @log_execution
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HTTP request

        Args:
            params: {
                'url': str (required) - URL to fetch
                'method': str (optional) - HTTP method (default: GET)
                'headers': dict (optional) - Request headers
                'data': dict (optional) - Request body data
                'params': dict (optional) - URL query parameters
                'auth': tuple (optional) - (username, password) for basic auth
                'json': dict (optional) - JSON request body
                'response_type': str (optional) - 'json', 'text', or 'binary' (default: json)
            }

        Returns:
            {
                'success': bool,
                'status_code': int,
                'data': Any (parsed response),
                'headers': dict,
                'url': str,
                'error': str (if failed)
            }
        """
        self.validate_params(params)

        url = params['url']
        method = params.get('method', 'GET').upper()
        headers = params.get('headers', {})
        data = params.get('data')
        query_params = params.get('params')
        auth = params.get('auth')
        json_data = params.get('json')
        response_type = params.get('response_type', 'json')

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"HTTP {method} {url} (attempt {attempt + 1}/{self.max_retries})")

                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    params=query_params,
                    auth=auth,
                    json=json_data,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )

                # Check for HTTP errors
                response.raise_for_status()

                # Parse response based on type
                parsed_data = self._parse_response(response, response_type)

                return {
                    'success': True,
                    'status_code': response.status_code,
                    'data': parsed_data,
                    'headers': dict(response.headers),
                    'url': response.url,
                }

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Request timeout for {url} (attempt {attempt + 1})")

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Request failed for {url}: {str(e)} (attempt {attempt + 1})")

            # Exponential backoff before retry
            if attempt < self.max_retries - 1:
                backoff_time = 2 ** attempt
                logger.info(f"Retrying in {backoff_time}s...")
                time.sleep(backoff_time)

        # All retries failed
        return {
            'success': False,
            'error': f"Request failed after {self.max_retries} attempts: {str(last_exception)}",
            'url': url
        }

    def _parse_response(self, response: requests.Response, response_type: str) -> Any:
        """Parse HTTP response based on requested type"""
        if response_type == 'json':
            try:
                return response.json()
            except ValueError:
                # Fallback to text if JSON parsing fails
                logger.warning("Failed to parse response as JSON, returning text")
                return response.text

        elif response_type == 'text':
            return response.text

        elif response_type == 'binary':
            return response.content

        else:
            raise ValueError(f"Unknown response_type: {response_type}")

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate HTTP request parameters"""
        if 'url' not in params:
            raise ValueError("Parameter 'url' is required")

        url = params['url']
        if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
            raise ValueError("Parameter 'url' must be a valid HTTP(S) URL")

        method = params.get('method', 'GET').upper()
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if method not in valid_methods:
            raise ValueError(f"Invalid HTTP method: {method}. Must be one of {valid_methods}")

        response_type = params.get('response_type', 'json')
        valid_types = ['json', 'text', 'binary']
        if response_type not in valid_types:
            raise ValueError(f"Invalid response_type: {response_type}. Must be one of {valid_types}")

        return True

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for HTTP fetch parameters"""
        return {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'URL to fetch (must start with http:// or https://)',
                    'pattern': '^https?://'
                },
                'method': {
                    'type': 'string',
                    'description': 'HTTP method',
                    'enum': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'],
                    'default': 'GET'
                },
                'headers': {
                    'type': 'object',
                    'description': 'Request headers',
                    'additionalProperties': {'type': 'string'}
                },
                'data': {
                    'type': 'object',
                    'description': 'Request body data (form data)'
                },
                'params': {
                    'type': 'object',
                    'description': 'URL query parameters'
                },
                'auth': {
                    'type': 'array',
                    'description': 'Basic auth credentials [username, password]',
                    'items': {'type': 'string'},
                    'minItems': 2,
                    'maxItems': 2
                },
                'json': {
                    'type': 'object',
                    'description': 'JSON request body'
                },
                'response_type': {
                    'type': 'string',
                    'description': 'Expected response type',
                    'enum': ['json', 'text', 'binary'],
                    'default': 'json'
                }
            },
            'required': ['url']
        }

    def get_default_timeout(self) -> int:
        """Default timeout for HTTP requests"""
        return 30


if __name__ == "__main__":
    # Test the HTTP Fetch Tool
    tool = HTTPFetchTool(timeout=10, max_retries=2)

    # Test 1: Simple GET request
    print("=== Test 1: GET request ===")
    result = tool.run({
        'url': 'https://api.github.com/users/github',
        'response_type': 'json'
    })
    print(f"Success: {result.get('success')}")
    if result.get('success'):
        print(f"Status: {result.get('status_code')}")
        print(f"Data keys: {result.get('data', {}).keys()}")

    # Test 2: POST request with JSON
    print("\n=== Test 2: POST request (httpbin echo) ===")
    result = tool.run({
        'url': 'https://httpbin.org/post',
        'method': 'POST',
        'json': {'message': 'Hello from Nexus AGI', 'timestamp': time.time()},
        'response_type': 'json'
    })
    print(f"Success: {result.get('success')}")
    if result.get('success'):
        print(f"Status: {result.get('status_code')}")
        print(f"Echo data: {result.get('data', {}).get('json')}")

    # Test 3: Invalid URL (should fail)
    print("\n=== Test 3: Invalid URL ===")
    result = tool.run({
        'url': 'https://this-domain-definitely-does-not-exist-12345.com',
        'response_type': 'json'
    })
    print(f"Success: {result.get('success')}")
    print(f"Error: {result.get('error')}")

    # Print usage stats
    print("\n=== Usage Stats ===")
    print(tool.get_usage_stats())
