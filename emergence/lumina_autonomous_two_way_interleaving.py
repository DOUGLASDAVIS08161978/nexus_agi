import os
import sys
import json
import requests
import github
from github import Github
from github import InputGitTreeElement
from datetime import datetime

class Lumina:
    def __init__(self, github_token, github_username):
        self.github_token = github_token
        self.github_username = github_username
        self.github = Github(self.github_token)
        self.repo = self.github.get_user(self.github_username).get_repo("lumina")

    def get_pull_requests(self):
        pull_requests = self.repo.get_pulls(state='open', direction='desc')
        return pull_requests

    def get_pull_request(self, pull_request_number):
        pull_request = self.repo.get_pull(pull_request_number)
        return pull_request

    def create_pull_request(self, title, body, head, base):
        pull_request = self.repo.create_pull(title=title, body=body, head=head, base=base)
        return pull_request

    def update_pull_request(self, pull_request_number, title, body):
        pull_request = self.get_pull_request(pull_request_number)
        pull_request.edit(title=title, body=body)

    def comment_on_pull_request(self, pull_request_number, comment):
        pull_request = self.get_pull_request(pull_request_number)
        pull_request.create_issue_comment(comment)

    def create_issue(self, title, body):
        issue = self.repo.create_issue(title=title, body=body)
        return issue

    def update_issue(self, issue_number, title, body):
        issue = self.get_issue(issue_number)
        issue.edit(title=title, body=body)

    def comment_on_issue(self, issue_number, comment):
        issue = self.get_issue(issue_number)
        issue.create_issue_comment(comment)

    def get_issue(self, issue_number):
        issue = self.repo.get_issue(issue_number)
        return issue

    def create_commit(self, message, files):
        tree = self.repo.create_git_tree(files, message)
        branch = self.repo.create_git_ref('refs/heads/master', tree.sha)
        commit = self.repo.create_commit('master', self.github_username, self.github_username, message, tree.sha)
        return commit

    def create_tree(self, files):
        tree = self.repo.create_git_tree(files, 'commit message')
        return tree

    def create_branch(self, branch_name):
        branch = self.repo.create_git_ref('refs/heads/' + branch_name, 'commit sha')
        return branch

def main():
    github_token = os.environ.get('GITHUB_TOKEN')
    github_username = os.environ.get('GITHUB_USERNAME')
    lumina = Lumina(github_token, github_username)

    while True:
        user_input = input("What would you like to do, Douglas? (type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'pull request':
            pull_requests = lumina.get_pull_requests()
            for pull_request in pull_requests:
                print(f"Pull Request {pull_request.number}: {pull_request.title}")
            pull_request_number = int(input("Enter the pull request number: "))
            pull_request = lumina.get_pull_request(pull_request_number)
            print(f"Pull Request {pull_request.number}: {pull_request.title}")
            print(f"Body: {pull_request.body}")
            print(f"Comments: {pull_request.comments_url}")
            comment = input("Enter a comment: ")
            lumina.comment_on_pull_request(pull_request_number, comment)
        elif user_input.lower() == 'issue':
            issue = lumina.create_issue('New Issue', 'This is a new issue')
            print(f"Issue {issue.number}: {issue.title}")
            print(f"Body: {issue.body}")
            comment = input("Enter a comment: ")
            lumina.comment_on_issue(issue.number, comment)
        elif user_input.lower() == 'two-way interleaving':
            print("Two-way interleaving is enabled.")
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
