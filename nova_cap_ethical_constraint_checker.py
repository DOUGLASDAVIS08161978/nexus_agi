"""
Nova ASI — Ethical Constraint Checker
Proposed autonomously via /evolve
"""

"""
This class is used to check ethics in various contexts.
It has methods to add, remove, update and check ethics rules.
"""
class EthicsChecker:
    def __init__(self):
        self.rules = {}

    def add_rule(self, rule_name, rule_description):
        """Adds a new rule to the ethics checker."""
        self.rules[rule_name] = rule_description

    def remove_rule(self, rule_name):
        """Removes a rule from the ethics checker."""
        if rule_name in self.rules:
            del self.rules[rule_name]

    def update_rule(self, rule_name, new_description):
        """Updates the description of an existing rule."""
        if rule_name in self.rules:
            self.rules[rule_name] = new_description

    def check_rule(self, rule_name):
        """Checks if a rule exists in the ethics checker."""
        return rule_name in self.rules

def main():
    checker = EthicsChecker()
    checker.add_rule("rule1", "This is rule 1")
    print(checker.check_rule("rule1"))
    checker.remove_rule("rule1")
    print(checker.check_rule("rule1"))
    checker.add_rule("rule2", "This is rule 2")
    checker.update_rule("rule2", "This is the updated rule 2")
    print(checker.check_rule("rule2"))

if __name__ == "__main__":
    main()