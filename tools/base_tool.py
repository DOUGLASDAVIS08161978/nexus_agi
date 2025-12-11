"""
Base Tool Class for Nexus AGI
Provides abstract interface for all external capability tools
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def log_execution(func):
    """Decorator to log tool execution"""
    @wraps(func)
    def wrapper(self, params: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        tool_name = self.__class__.__name__

        logger.info(f"[{tool_name}] Starting execution with params: {params}")

        try:
            result = func(self, params)
            execution_time = time.time() - start_time

            logger.info(f"[{tool_name}] Completed successfully in {execution_time:.2f}s")

            # Add metadata to result
            if isinstance(result, dict):
                result['_metadata'] = {
                    'tool': tool_name,
                    'execution_time': execution_time,
                    'timestamp': datetime.utcnow().isoformat(),
                    'success': True
                }

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[{tool_name}] Failed after {execution_time:.2f}s: {str(e)}")

            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                '_metadata': {
                    'tool': tool_name,
                    'execution_time': execution_time,
                    'timestamp': datetime.utcnow().isoformat(),
                    'success': False
                }
            }

    return wrapper


class BaseTool(ABC):
    """
    Abstract base class for all Nexus AGI tools

    All tools must implement the run() method with a clean dict interface:
    - Input: params (Dict[str, Any])
    - Output: result (Dict[str, Any])
    """

    def __init__(
        self,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        rate_limit: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize base tool

        Args:
            timeout: Maximum execution time in seconds
            max_retries: Maximum number of retry attempts
            rate_limit: Maximum requests per minute
            **kwargs: Additional tool-specific configuration
        """
        self.timeout = timeout or self.get_default_timeout()
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.config = kwargs

        # Usage tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_execution_time = 0.0

        logger.info(f"Initialized {self.__class__.__name__} with timeout={self.timeout}s")

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with given parameters

        Args:
            params: Dictionary of input parameters (tool-specific)

        Returns:
            Dictionary containing:
                - success: bool (True if successful)
                - data: Any (tool-specific output data)
                - error: Optional[str] (error message if failed)
                - _metadata: Dict (execution metadata)
        """
        pass

    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate input parameters

        Args:
            params: Input parameters to validate

        Returns:
            True if valid, raises ValueError if invalid
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool parameters

        Returns:
            JSON schema describing expected parameters
        """
        pass

    def get_default_timeout(self) -> int:
        """Get default timeout for this tool"""
        return 30  # 30 seconds default

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool"""
        return {
            'tool': self.__class__.__name__,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            'average_execution_time': self.total_execution_time / self.total_calls if self.total_calls > 0 else 0,
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(timeout={self.timeout}s)"


class ToolRegistry:
    """
    Registry for managing and executing tools

    Provides:
    - Tool registration and discovery
    - Centralized execution with logging
    - Usage tracking and limits
    - Error handling
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        logger.info("Initialized ToolRegistry")

    def register(self, name: str, tool: BaseTool) -> None:
        """
        Register a tool

        Args:
            name: Unique identifier for the tool
            tool: BaseTool instance
        """
        if name in self._tools:
            logger.warning(f"Overwriting existing tool: {name}")

        self._tools[name] = tool
        logger.info(f"Registered tool: {name} ({tool.__class__.__name__})")

    def unregister(self, name: str) -> None:
        """Unregister a tool"""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a registered tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())

    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get information about a registered tool"""
        tool = self.get_tool(name)
        if not tool:
            return {'error': f'Tool not found: {name}'}

        return {
            'name': name,
            'class': tool.__class__.__name__,
            'timeout': tool.timeout,
            'max_retries': tool.max_retries,
            'rate_limit': tool.rate_limit,
            'schema': tool.get_schema(),
            'usage_stats': tool.get_usage_stats(),
        }

    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered tool

        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool

        Returns:
            Tool execution result
        """
        tool = self.get_tool(tool_name)

        if not tool:
            return {
                'success': False,
                'error': f'Tool not found: {tool_name}',
                'available_tools': self.list_tools()
            }

        try:
            # Validate parameters
            tool.validate_params(params)

            # Execute tool
            result = tool.run(params)

            # Update statistics
            tool.total_calls += 1
            if result.get('success', False):
                tool.successful_calls += 1
            else:
                tool.failed_calls += 1

            if '_metadata' in result:
                tool.total_execution_time += result['_metadata'].get('execution_time', 0)

            return result

        except ValueError as e:
            logger.error(f"Parameter validation failed for {tool_name}: {str(e)}")
            return {
                'success': False,
                'error': f'Invalid parameters: {str(e)}',
                'schema': tool.get_schema()
            }

        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {str(e)}")
            tool.total_calls += 1
            tool.failed_calls += 1
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all registered tools"""
        return {
            'total_tools': len(self._tools),
            'tools': {
                name: tool.get_usage_stats()
                for name, tool in self._tools.items()
            }
        }


if __name__ == "__main__":
    # Example usage
    class ExampleTool(BaseTool):
        """Example tool for testing"""

        @log_execution
        def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
            self.validate_params(params)
            return {
                'success': True,
                'data': {'message': f"Hello {params.get('name', 'World')}!"}
            }

        def validate_params(self, params: Dict[str, Any]) -> bool:
            if 'name' not in params:
                raise ValueError("Parameter 'name' is required")
            return True

        def get_schema(self) -> Dict[str, Any]:
            return {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Name to greet'}
                },
                'required': ['name']
            }

    # Test the registry
    registry = ToolRegistry()
    registry.register('example', ExampleTool())

    print("Available tools:", registry.list_tools())
    print("\nTool info:", registry.get_tool_info('example'))

    result = registry.execute('example', {'name': 'Nexus AGI'})
    print("\nExecution result:", result)

    print("\nUsage stats:", registry.get_all_stats())
