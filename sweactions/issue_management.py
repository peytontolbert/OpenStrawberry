import os
import logging
from swe.swe_actions import Action, Context

class CreateIssueAction(Action):
    def execute(self, context: Context):
        """
        Creates an issue in project management tools like GitHub Issues or Jira.
        """
        issue_title = self.extract_issue_title(context.response)
        issue_description = self.extract_issue_description(context.response)
        # Placeholder: Integration with GitHub API or Jira API
        # Example using GitHub API
        import requests

        github_token = os.getenv('GITHUB_TOKEN')
        repo = context.entities.get('repo', 'user/repo')
        api_url = f"https://api.github.com/repos/{repo}/issues"

        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        data = {
            "title": issue_title,
            "body": issue_description
        }

        try:
            response = requests.post(api_url, json=data, headers=headers)
            if response.status_code == 201:
                print(f"Issue '{issue_title}' created successfully.")
                logging.info(f"Issue '{issue_title}' created successfully.")
            else:
                print(f"Failed to create issue: {response.json()}")
                logging.error(f"Failed to create issue: {response.json()}")
        except Exception as e:
            error_msg = f"Error creating issue: {e}"
            print(error_msg)
            logging.error(error_msg)


class AssignTaskAction(Action):
    def execute(self, context: Context):
        """
        Assigns tasks to team members in project management tools.
        """
        # Placeholder: Integration with project management tools API
        assignee = context.entities.get('assignee')
        issue_number = context.entities.get('issue_number')
        if not assignee or not issue_number:
            print("Assignee or issue number not provided.")
            logging.warning("Assignee or issue number not provided.")
            return

        import requests

        github_token = os.getenv('GITHUB_TOKEN')
        repo = context.entities.get('repo', 'user/repo')
        api_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        data = {
            "assignees": [assignee]
        }

        try:
            response = requests.patch(api_url, json=data, headers=headers)
            if response.status_code == 200:
                print(f"Issue #{issue_number} assigned to {assignee}.")
                logging.info(f"Issue #{issue_number} assigned to {assignee}.")
            else:
                print(f"Failed to assign issue: {response.json()}")
                logging.error(f"Failed to assign issue: {response.json()}")
        except Exception as e:
            error_msg = f"Error assigning task: {e}"
            print(error_msg)
            logging.error(error_msg)
