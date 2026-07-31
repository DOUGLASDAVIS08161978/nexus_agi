# error_analyzer.py

import logging
import traceback
import ast
import inspect
from typing import List, Dict

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    def __init__(self):
        self.error_messages = {}
        self.error_counts = {}

    def analyze_error(self, error_message: str, error_type: str, error_line: int, error_column: int):
        """
        Analyze an error and store it in the error_messages and error_counts dictionaries.

        Args:
        - error_message (str): The error message.
        - error_type (str): The type of error (e.g., SyntaxError, TypeError, etc.).
        - error_line (int): The line number where the error occurred.
        - error_column (int): The column number where the error occurred.
        """
        if error_type not in self.error_messages:
            self.error_messages[error_type] = []
            self.error_counts[error_type] = 0

        error_dict = {
            "message": error_message,
            "type": error_type,
            "line": error_line,
            "column": error_column
        }
        self.error_messages[error_type].append(error_dict)
        self.error_counts[error_type] += 1

    def print_error_messages(self):
        """
        Print the error messages and counts for each error type.
        """
        for error_type, error_dicts in self.error_messages.items():
            logger.info(f"Error Type: {error_type}")
            logger.info(f"Error Count: {self.error_counts[error_type]}")
            for error_dict in error_dicts:
                logger.info(f"Message: {error_dict['message']}")
                logger.info(f"Line: {error_dict['line']}")
                logger.info(f"Column: {error_dict['column']}")
                logger.info("-" * 50)

    def analyze_code(self, code: str):
        """
        Analyze the given code for errors.

        Args:
        - code (str): The code to analyze.
        """
        try:
            # Try to parse the code
            tree = ast.parse(code)
        except SyntaxError as e:
            # If a syntax error occurs, analyze it
            self.analyze_error(str(e), "SyntaxError", e.lineno, e.offset)
        except Exception as e:
            # If any other error occurs, analyze it
            self.analyze_error(str(e), "RuntimeError", 0, 0)

    def analyze_function(self, func: callable):
        """
        Analyze the given function for errors.

        Args:
        - func (callable): The function to analyze.
        """
        try:
            # Try to get the function's source code
            source = inspect.getsource(func)
        except Exception as e:
            # If any error occurs, analyze it
            self.analyze_error(str(e), "RuntimeError", 0, 0)
            return

        # Analyze the function's source code
        self.analyze_code(source)


# Example usage
if __name__ == "__main__":
    analyzer = ErrorAnalyzer()

    # Analyze some sample code
    code = """
x = 5
y = "hello"
print(x + y)
"""
    analyzer.analyze_code(code)

    # Analyze a function
    def example_function():
        x = 5
        y = "hello"
        print(x + y)

    analyzer.analyze_function(example_function)

    # Print the error messages
    analyzer.print_error_messages()
This code defines an `ErrorAnalyzer` class that can analyze errors in code and functions. It uses the `ast` module to parse code and the `inspect` module to get function source code. The `analyze_error` method stores error messages and counts in dictionaries, and the `print_error_messages` method prints these dictionaries. The `analyze_code` method tries to parse the given code and analyzes any errors that occur, while the `analyze_function` method gets a function's source code and analyzes it.
