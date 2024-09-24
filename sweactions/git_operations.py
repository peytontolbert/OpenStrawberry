from swe_actions import Action, Context
import os
import logging
import subprocess


class CommitChangesAction(Action):
    def execute(self, context: Context):
        """
        Commits the current changes to Git with the given message.
        """
        message = context.entities.get('commit_message')
        if not message:
            message = self.extract_commit_message(context.response)
        if message:
            try:
                subprocess.run(["git", "add", "."], cwd=context.project_directory)
                subprocess.run(["git", "commit", "-m", message], cwd=context.project_directory)
                print("Changes committed to Git.")
                logging.info("Changes committed to Git.")
            except Exception as e:
                error_msg = f"Error committing changes to Git: {e}"
                print(error_msg)
                logging.error(error_msg)
          

class CreateBranchAction(Action):
    def execute(self, context: Context):
        """
        Creates a new Git branch.
        """
        branch_name = context.entities.get('branch_name')
        if not branch_name:
            branch_name = self.extract_branch_name(context.response)
        if branch_name:
            try:
                subprocess.run(["git", "checkout", "-b", branch_name], cwd=context.project_directory, check=True)
                print(f"Created and switched to branch '{branch_name}'.")
                logging.info(f"Created and switched to branch '{branch_name}'.")
                context.agent.update_state()
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to create branch '{branch_name}': {e}"
                print(error_msg)
                logging.error(error_msg)
        else:
            error_msg = "Could not extract branch name."
            print(error_msg)
            logging.error(error_msg)



class MergeBranchAction(Action):
    def execute(self, context: Context):
        """
        Merges one Git branch into another.
        """
        target_branch = context.entities.get('target_branch')
        if not target_branch:
            target_branch = self.extract_merge_target_branch(context.response)
        if target_branch:
            try:
                subprocess.run(["git", "checkout", target_branch], cwd=context.project_directory, check=True)
                subprocess.run(["git", "merge", context.entities.get('source_branch', 'feature')], cwd=context.project_directory, check=True)
                print(f"Merged branch into '{target_branch}'.")
                logging.info(f"Merged branch into '{target_branch}'.")
                context.agent.update_state()
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to merge branch into '{target_branch}': {e}"
                print(error_msg)
                logging.error(error_msg)
        else:
            error_msg = "Could not extract target branch name."
            print(error_msg)
            logging.error(error_msg)


class PushChangesAction(Action):
    def execute(self, context: Context):
        """
        Pushes committed changes to the remote repository.
        """
        try:
            subprocess.run(["git", "push"], cwd=context.project_directory, check=True)
            print("Changes pushed to remote repository.")
            logging.info("Changes pushed to remote repository.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to push changes: {e}"
            print(error_msg)
            logging.error(error_msg)


class PullChangesAction(Action):
    def execute(self, context: Context):
        """
        Pulls the latest changes from the remote repository.
        """
        try:
            subprocess.run(["git", "pull"], cwd=context.project_directory, check=True)
            print("Pulled latest changes from remote repository.")
            logging.info("Pulled latest changes from remote repository.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to pull changes: {e}"
            print(error_msg)
            logging.error(error_msg)


class ResolveMergeConflictsAction(Action):
    def execute(self, context: Context):
        """
        Attempts to automatically resolve Git merge conflicts.
        """
        try:
            # Simple strategy: accept current changes
            subprocess.run(["git", "merge", "--strategy-option=theirs"], cwd=context.project_directory, check=True)
            print("Merge conflicts resolved using 'theirs' strategy.")
            logging.info("Merge conflicts resolved using 'theirs' strategy.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to resolve merge conflicts: {e}"
            print(error_msg)
            logging.error(error_msg)


class RevertChangesAction(Action):
    def execute(self, context: Context):
        """
        Reverts commits or changes in the repository.
        """
        revert_target = context.entities.get('revert_target', 'HEAD~1')  # Default to last commit
        try:
            subprocess.run(["git", "revert", revert_target, "--no-edit"], cwd=context.project_directory, check=True)
            print(f"Reverted changes up to {revert_target}.")
            logging.info(f"Reverted changes up to {revert_target}.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to revert changes: {e}"
            print(error_msg)
            logging.error(error_msg)

