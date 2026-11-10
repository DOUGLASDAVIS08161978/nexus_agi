# self_modification.py

import ast
import importlib
import inspect
import os
import sys

class SelfModification:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = importlib.import_module(module_name)
        self.modified_code = {}

    def get_module_code(self):
        return inspect.getsource(self.module)

    def parse_module_code(self):
        return ast.parse(self.get_module_code())

    def modify_code(self, node, modification):
        if isinstance(node, ast.Module):
            for child in node.body:
                self.modify_code(child, modification)
        elif isinstance(node, ast.FunctionDef):
            if modification == 'add_print':
                node.body.insert(0, ast.Expr(ast.Call(ast.Name('print', ast.Load()), [], None)))
        elif isinstance(node, ast.ClassDef):
            if modification == 'add_method':
                node.body.append(ast.FunctionDef('new_method', [], [], None, []))

    def apply_modifications(self, modifications):
        tree = self.parse_module_code()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                for modification in modifications:
                    self.modify_code(node, modification)
        self.modified_code = ast.unparse(tree)

    def save_modified_code(self):
        with open(self.module_name + '.py', 'w') as f:
            f.write(self.modified_code)

    def modify_module(self, modifications):
        self.apply_modifications(modifications)
        self.save_modified_code()

def main():
    modification_module = SelfModification('modification_module')
    modifications = ['add_print', 'add_method']
    modification_module.modify_module(modifications)

if __name__ == '__main__':
    main()
