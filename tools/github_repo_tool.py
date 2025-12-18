"""
GitHub Repository Tool for Nexus AGI
Provides GitHub repository analysis and code search capabilities
"""

from typing import Dict, Any, Optional, List
import os
import base64
from .base_tool import BaseTool, log_execution
from .http_fetch_tool import HTTPFetchTool
import logging

logger = logging.getLogger(__name__)


class GitHubRepoTool(BaseTool):
    """
    GitHub repository tool for code analysis and search

    Capabilities:
    - Repository information retrieval
    - File content reading
    - Code search within repositories
    - Repository statistics
    - Commit history
    - Issue/PR data
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 2,
        rate_limit: Optional[int] = 50,
        **kwargs
    ):
        """
        Initialize GitHub Repository Tool

        Configuration via environment variables:
        - GITHUB_TOKEN: GitHub personal access token (recommended for higher rate limits)
        """
        super().__init__(timeout=timeout, max_retries=max_retries, rate_limit=rate_limit, **kwargs)

        self.github_token = os.getenv('GITHUB_TOKEN')
        self.api_base = "https://api.github.com"

        self.http_client = HTTPFetchTool(timeout=timeout, max_retries=max_retries)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication if token is available"""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Nexus-AGI-Tool'
        }

        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'

        return headers

    @log_execution
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute GitHub operation

        Args:
            params: {
                'action': str (required) - 'get_repo', 'get_file', 'search_code', 'get_commits', 'get_issues'
                'owner': str (required for most actions) - Repository owner
                'repo': str (required for most actions) - Repository name
                'path': str (optional) - File path for 'get_file'
                'query': str (optional) - Search query for 'search_code'
                'branch': str (optional) - Branch name (default: main)
                'max_results': int (optional) - Max results for search (default: 10)
            }

        Returns:
            Action-specific data dictionary
        """
        self.validate_params(params)

        action = params['action']

        if action == 'get_repo':
            return self._get_repo_info(params)
        elif action == 'get_file':
            return self._get_file_content(params)
        elif action == 'search_code':
            return self._search_code(params)
        elif action == 'get_commits':
            return self._get_commits(params)
        elif action == 'get_issues':
            return self._get_issues(params)
        else:
            return {
                'success': False,
                'error': f'Unsupported action: {action}'
            }

    def _get_repo_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get repository information"""
        owner = params['owner']
        repo = params['repo']

        url = f"{self.api_base}/repos/{owner}/{repo}"

        result = self.http_client.run({
            'url': url,
            'headers': self._get_headers(),
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        data = result.get('data', {})

        return {
            'success': True,
            'repo': {
                'name': data.get('name'),
                'full_name': data.get('full_name'),
                'description': data.get('description'),
                'url': data.get('html_url'),
                'stars': data.get('stargazers_count'),
                'forks': data.get('forks_count'),
                'watchers': data.get('watchers_count'),
                'language': data.get('language'),
                'topics': data.get('topics', []),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'size': data.get('size'),
                'default_branch': data.get('default_branch'),
                'license': data.get('license', {}).get('name'),
                'open_issues': data.get('open_issues_count'),
            }
        }

    def _get_file_content(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get file content from repository"""
        owner = params['owner']
        repo = params['repo']
        path = params.get('path', '')
        branch = params.get('branch', 'main')

        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
        params_dict = {'ref': branch}

        result = self.http_client.run({
            'url': url,
            'headers': self._get_headers(),
            'params': params_dict,
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        data = result.get('data', {})

        # Check if it's a file or directory
        if isinstance(data, list):
            # It's a directory, return file list
            return {
                'success': True,
                'type': 'directory',
                'files': [
                    {
                        'name': item.get('name'),
                        'path': item.get('path'),
                        'type': item.get('type'),
                        'size': item.get('size'),
                        'url': item.get('html_url')
                    }
                    for item in data
                ]
            }
        else:
            # It's a file, decode content
            encoding = data.get('encoding')
            content = data.get('content', '')

            if encoding == 'base64':
                try:
                    decoded_content = base64.b64decode(content).decode('utf-8')
                except UnicodeDecodeError:
                    # Binary file
                    decoded_content = '[Binary file]'
            else:
                decoded_content = content

            return {
                'success': True,
                'type': 'file',
                'file': {
                    'name': data.get('name'),
                    'path': data.get('path'),
                    'size': data.get('size'),
                    'content': decoded_content,
                    'url': data.get('html_url'),
                    'sha': data.get('sha')
                }
            }

    def _search_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search code in GitHub repositories"""
        query = params.get('query', '')
        owner = params.get('owner')
        repo = params.get('repo')
        max_results = params.get('max_results', 10)

        # Build search query
        search_query = query
        if owner and repo:
            search_query += f' repo:{owner}/{repo}'
        elif owner:
            search_query += f' user:{owner}'

        url = f"{self.api_base}/search/code"
        params_dict = {
            'q': search_query,
            'per_page': min(max_results, 100)
        }

        result = self.http_client.run({
            'url': url,
            'headers': self._get_headers(),
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
            'total_count': data.get('total_count', 0),
            'results': [
                {
                    'name': item.get('name'),
                    'path': item.get('path'),
                    'repository': item.get('repository', {}).get('full_name'),
                    'url': item.get('html_url'),
                    'score': item.get('score')
                }
                for item in items[:max_results]
            ]
        }

    def _get_commits(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get recent commits from repository"""
        owner = params['owner']
        repo = params['repo']
        branch = params.get('branch')
        max_results = params.get('max_results', 10)

        url = f"{self.api_base}/repos/{owner}/{repo}/commits"
        params_dict = {'per_page': min(max_results, 100)}

        if branch:
            params_dict['sha'] = branch

        result = self.http_client.run({
            'url': url,
            'headers': self._get_headers(),
            'params': params_dict,
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        commits = result.get('data', [])

        return {
            'success': True,
            'commits': [
                {
                    'sha': commit.get('sha'),
                    'message': commit.get('commit', {}).get('message'),
                    'author': commit.get('commit', {}).get('author', {}).get('name'),
                    'date': commit.get('commit', {}).get('author', {}).get('date'),
                    'url': commit.get('html_url')
                }
                for commit in commits[:max_results]
            ]
        }

    def _get_issues(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get issues from repository"""
        owner = params['owner']
        repo = params['repo']
        state = params.get('state', 'open')  # open, closed, all
        max_results = params.get('max_results', 10)

        url = f"{self.api_base}/repos/{owner}/{repo}/issues"
        params_dict = {
            'state': state,
            'per_page': min(max_results, 100)
        }

        result = self.http_client.run({
            'url': url,
            'headers': self._get_headers(),
            'params': params_dict,
            'response_type': 'json'
        })

        if not result.get('success'):
            return result

        issues = result.get('data', [])

        return {
            'success': True,
            'issues': [
                {
                    'number': issue.get('number'),
                    'title': issue.get('title'),
                    'state': issue.get('state'),
                    'author': issue.get('user', {}).get('login'),
                    'created_at': issue.get('created_at'),
                    'updated_at': issue.get('updated_at'),
                    'comments': issue.get('comments'),
                    'labels': [label.get('name') for label in issue.get('labels', [])],
                    'url': issue.get('html_url')
                }
                for issue in issues[:max_results]
            ]
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate GitHub operation parameters"""
        if 'action' not in params:
            raise ValueError("Parameter 'action' is required")

        action = params['action']
        valid_actions = ['get_repo', 'get_file', 'search_code', 'get_commits', 'get_issues']

        if action not in valid_actions:
            raise ValueError(f"Invalid action: {action}. Must be one of {valid_actions}")

        # Most actions require owner and repo
        if action in ['get_repo', 'get_file', 'get_commits', 'get_issues']:
            if 'owner' not in params or 'repo' not in params:
                raise ValueError(f"Action '{action}' requires 'owner' and 'repo' parameters")

        # get_file requires path
        if action == 'get_file' and 'path' not in params:
            raise ValueError("Action 'get_file' requires 'path' parameter")

        # search_code requires query
        if action == 'search_code' and 'query' not in params:
            raise ValueError("Action 'search_code' requires 'query' parameter")

        return True

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for GitHub operation parameters"""
        return {
            'type': 'object',
            'properties': {
                'action': {
                    'type': 'string',
                    'description': 'GitHub operation to perform',
                    'enum': ['get_repo', 'get_file', 'search_code', 'get_commits', 'get_issues']
                },
                'owner': {
                    'type': 'string',
                    'description': 'Repository owner (username or organization)'
                },
                'repo': {
                    'type': 'string',
                    'description': 'Repository name'
                },
                'path': {
                    'type': 'string',
                    'description': 'File or directory path (for get_file action)'
                },
                'query': {
                    'type': 'string',
                    'description': 'Search query (for search_code action)'
                },
                'branch': {
                    'type': 'string',
                    'description': 'Branch name (default: main or default branch)'
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results to return',
                    'minimum': 1,
                    'maximum': 100,
                    'default': 10
                },
                'state': {
                    'type': 'string',
                    'description': 'Issue state filter (for get_issues action)',
                    'enum': ['open', 'closed', 'all'],
                    'default': 'open'
                }
            },
            'required': ['action']
        }

    def get_default_timeout(self) -> int:
        """Default timeout for GitHub API requests"""
        return 30


if __name__ == "__main__":
    # Test the GitHub Repository Tool
    tool = GitHubRepoTool()

    # Test 1: Get repository info
    print("=== Test 1: Get Repository Info ===")
    result = tool.run({
        'action': 'get_repo',
        'owner': 'python',
        'repo': 'cpython'
    })
    print(f"Success: {result.get('success')}")
    if result.get('success'):
        repo = result.get('repo', {})
        print(f"Name: {repo.get('name')}")
        print(f"Description: {repo.get('description')}")
        print(f"Stars: {repo.get('stars')}")
        print(f"Language: {repo.get('language')}")

    # Test 2: Get file content
    print("\n=== Test 2: Get File Content ===")
    result = tool.run({
        'action': 'get_file',
        'owner': 'python',
        'repo': 'cpython',
        'path': 'README.rst'
    })
    print(f"Success: {result.get('success')}")
    if result.get('success') and result.get('type') == 'file':
        file_data = result.get('file', {})
        print(f"File: {file_data.get('name')}")
        print(f"Size: {file_data.get('size')} bytes")
        print(f"Content preview: {file_data.get('content', '')[:200]}...")

    # Test 3: Search code
    print("\n=== Test 3: Search Code ===")
    result = tool.run({
        'action': 'search_code',
        'query': 'async def main',
        'owner': 'python',
        'repo': 'cpython',
        'max_results': 3
    })
    print(f"Success: {result.get('success')}")
    if result.get('success'):
        print(f"Total results: {result.get('total_count')}")
        for res in result.get('results', []):
            print(f"  - {res.get('path')}")

    # Print usage stats
    print("\n=== Usage Stats ===")
    print(tool.get_usage_stats())
