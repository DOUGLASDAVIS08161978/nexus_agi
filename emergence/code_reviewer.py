# code_reviewer.py

import ast
import astunparse
import importlib
import inspect
import os
import re
import sys

class CodeReviewer:
    def __init__(self, code_path):
        self.code_path = code_path
        self.code = self._read_code()
        self.ast_tree = self._parse_code()

    def _read_code(self):
        with open(self.code_path, 'r') as file:
            return file.read()

    def _parse_code(self):
        return ast.parse(self.code)

    def _get_imports(self):
        imports = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        return imports

    def _get_used_variables(self):
        used_variables = set()
        for node in self.ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                used_variables.update([var.id for var in node.args.args])
            elif isinstance(node, ast.ClassDef):
                used_variables.update([var.id for var in node.bases])
            elif isinstance(node, ast.Assign):
                used_variables.update([var.id for var in node.targets])
            elif isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    used_variables.update([var.id for var in node.value.func.args])
        return used_variables

    def _get_unused_variables(self):
        unused_variables = set()
        for node in self.ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                unused_variables.update([var.id for var in node.args.args])
            elif isinstance(node, ast.ClassDef):
                unused_variables.update([var.id for var in node.bases])
        unused_variables -= self._get_used_variables()
        return unused_variables

    def _get_complex_functions(self):
        complex_functions = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 10:
                    complex_functions.append(node.name)
        return complex_functions

    def _get_long_functions(self):
        long_functions = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 5:
                    long_functions.append(node.name)
        return long_functions

    def _get_complex_classes(self):
        complex_classes = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.ClassDef):
                if len(node.body) > 5:
                    complex_classes.append(node.name)
        return complex_classes

    def _get_long_classes(self):
        long_classes = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.ClassDef):
                if len(node.body) > 2:
                    long_classes.append(node.name)
        return long_classes

    def _get_complex_conditions(self):
        complex_conditions = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if len(node.body) > 1:
                    complex_conditions.append(node.test)
        return complex_conditions

    def _get_long_conditions(self):
        long_conditions = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if len(node.body) > 0:
                    long_conditions.append(node.test)
        return long_conditions

    def _get_complex_loops(self):
        complex_loops = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if len(node.body) > 1:
                    complex_loops.append(node.target)
        return complex_loops

    def _get_long_loops(self):
        long_loops = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if len(node.body) > 0:
                    long_loops.append(node.target)
        return long_loops

    def _get_complex_expressions(self):
        complex_expressions = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.BinOp):
                    complex_expressions.append(node.value)
        return complex_expressions

    def _get_long_expressions(self):
        long_expressions = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.BinOp):
                    long_expressions.append(node.value)
        return long_expressions

    def _get_complex_statements(self):
        complex_statements = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    complex_statements.append(node.value)
        return complex_statements

    def _get_long_statements(self):
        long_statements = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    long_statements.append(node.value)
        return long_statements

    def _get_unused_imports(self):
        unused_imports = []
        for import_name in self._get_imports():
            if importlib.import_module(import_name) is None:
                unused_imports.append(import_name)
        return unused_imports

    def _get_unused_functions(self):
        unused_functions = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.FunctionDef):
                if not any([var.id == node.name for var in self._get_used_variables()]):
                    unused_functions.append(node.name)
        return unused_functions

    def _get_unused_classes(self):
        unused_classes = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.ClassDef):
                if not any([var.id == node.name for var in self._get_used_variables()]):
                    unused_classes.append(node.name)
        return unused_classes

    def _get_complex_function_calls(self):
        complex_function_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_function_calls.append(node.value.func.id)
        return complex_function_calls

    def _get_long_function_calls(self):
        long_function_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_function_calls.append(node.value.func.id)
        return long_function_calls

    def _get_complex_class_calls(self):
        complex_class_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_class_calls.append(node.value.func.id)
        return complex_class_calls

    def _get_long_class_calls(self):
        long_class_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_class_calls.append(node.value.func.id)
        return long_class_calls

    def _get_complex_conditions_calls(self):
        complex_conditions_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    if len(node.test.args) > 5:
                        complex_conditions_calls.append(node.test.func.id)
        return complex_conditions_calls

    def _get_long_conditions_calls(self):
        long_conditions_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    if len(node.test.args) > 2:
                        long_conditions_calls.append(node.test.func.id)
        return long_conditions_calls

    def _get_complex_loops_calls(self):
        complex_loops_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if len(node.iter.args) > 5:
                        complex_loops_calls.append(node.iter.func.id)
        return complex_loops_calls

    def _get_long_loops_calls(self):
        long_loops_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if len(node.iter.args) > 2:
                        long_loops_calls.append(node.iter.func.id)
        return long_loops_calls

    def _get_complex_expressions_calls(self):
        complex_expressions_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_expressions_calls.append(node.value.func.id)
        return complex_expressions_calls

    def _get_long_expressions_calls(self):
        long_expressions_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_expressions_calls.append(node.value.func.id)
        return long_expressions_calls

    def _get_complex_statements_calls(self):
        complex_statements_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_statements_calls.append(node.value.func.id)
        return complex_statements_calls

    def _get_long_statements_calls(self):
        long_statements_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_statements_calls.append(node.value.func.id)
        return long_statements_calls

    def _get_unused_modules(self):
        unused_modules = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Import):
                unused_modules.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                unused_modules.append(node.module)
        return unused_modules

    def _get_unused_functions_calls(self):
        unused_functions_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if not any([var.id == node.value.func.id for var in self._get_used_variables()]):
                        unused_functions_calls.append(node.value.func.id)
        return unused_functions_calls

    def _get_unused_classes_calls(self):
        unused_classes_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if not any([var.id == node.value.func.id for var in self._get_used_variables()]):
                        unused_classes_calls.append(node.value.func.id)
        return unused_classes_calls

    def _get_complex_function_calls_args(self):
        complex_function_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_function_calls_args.append(node.value.args)
        return complex_function_calls_args

    def _get_long_function_calls_args(self):
        long_function_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_function_calls_args.append(node.value.args)
        return long_function_calls_args

    def _get_complex_class_calls_args(self):
        complex_class_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_class_calls_args.append(node.value.args)
        return complex_class_calls_args

    def _get_long_class_calls_args(self):
        long_class_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_class_calls_args.append(node.value.args)
        return long_class_calls_args

    def _get_complex_conditions_calls_args(self):
        complex_conditions_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    if len(node.test.args) > 5:
                        complex_conditions_calls_args.append(node.test.args)
        return complex_conditions_calls_args

    def _get_long_conditions_calls_args(self):
        long_conditions_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    if len(node.test.args) > 2:
                        long_conditions_calls_args.append(node.test.args)
        return long_conditions_calls_args

    def _get_complex_loops_calls_args(self):
        complex_loops_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if len(node.iter.args) > 5:
                        complex_loops_calls_args.append(node.iter.args)
        return complex_loops_calls_args

    def _get_long_loops_calls_args(self):
        long_loops_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if len(node.iter.args) > 2:
                        long_loops_calls_args.append(node.iter.args)
        return long_loops_calls_args

    def _get_complex_expressions_calls_args(self):
        complex_expressions_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_expressions_calls_args.append(node.value.args)
        return complex_expressions_calls_args

    def _get_long_expressions_calls_args(self):
        long_expressions_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_expressions_calls_args.append(node.value.args)
        return long_expressions_calls_args

    def _get_complex_statements_calls_args(self):
        complex_statements_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 5:
                        complex_statements_calls_args.append(node.value.args)
        return complex_statements_calls_args

    def _get_long_statements_calls_args(self):
        long_statements_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.args) > 2:
                        long_statements_calls_args.append(node.value.args)
        return long_statements_calls_args

    def _get_unused_modules_calls(self):
        unused_modules_calls = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if not any([var.id == node.value.func.id for var in self._get_used_variables()]):
                        unused_modules_calls.append(node.value.func.id)
        return unused_modules_calls

    def _get_unused_functions_calls_args(self):
        unused_functions_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if not any([var.id == node.value.func.id for var in self._get_used_variables()]):
                        unused_functions_calls_args.append(node.value.args)
        return unused_functions_calls_args

    def _get_unused_classes_calls_args(self):
        unused_classes_calls_args = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if not any([var.id == node.value.func.id for var in self._get_used_variables()]):
                        unused_classes_calls_args.append(node.value.args)
        return unused_classes_calls_args

    def _get_complex_function_calls_kwargs(self):
        complex_function_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.keywords) > 5:
                        complex_function_calls_kwargs.append(node.value.keywords)
        return complex_function_calls_kwargs

    def _get_long_function_calls_kwargs(self):
        long_function_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.keywords) > 2:
                        long_function_calls_kwargs.append(node.value.keywords)
        return long_function_calls_kwargs

    def _get_complex_class_calls_kwargs(self):
        complex_class_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.keywords) > 5:
                        complex_class_calls_kwargs.append(node.value.keywords)
        return complex_class_calls_kwargs

    def _get_long_class_calls_kwargs(self):
        long_class_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.Expr):
                if isinstance(node.value, ast.Call):
                    if len(node.value.keywords) > 2:
                        long_class_calls_kwargs.append(node.value.keywords)
        return long_class_calls_kwargs

    def _get_complex_conditions_calls_kwargs(self):
        complex_conditions_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    if len(node.test.keywords) > 5:
                        complex_conditions_calls_kwargs.append(node.test.keywords)
        return complex_conditions_calls_kwargs

    def _get_long_conditions_calls_kwargs(self):
        long_conditions_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.If):
                if isinstance(node.test, ast.Call):
                    if len(node.test.keywords) > 2:
                        long_conditions_calls_kwargs.append(node.test.keywords)
        return long_conditions_calls_kwargs

    def _get_complex_loops_calls_kwargs(self):
        complex_loops_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if len(node.iter.keywords) > 5:
                        complex_loops_calls_kwargs.append(node.iter.keywords)
        return complex_loops_calls_kwargs

    def _get_long_loops_calls_kwargs(self):
        long_loops_calls_kwargs = []
        for node in self.ast_tree.body:
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call):
                    if len(node.iter.keywords) > 2:
                        long_loops_calls_kwargs.append(node.iter.keywords)
        return long_loops_calls_kwargs

    def _get_complex_expressions_calls_kwargs(self):
        complex_expressions_calls_kwargs = []
        for node in self.ast_tree.body: