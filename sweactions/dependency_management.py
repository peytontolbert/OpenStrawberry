import subprocess
import logging
from swe.swe_actions import Action, Context

class ManageDependenciesAction(Action):
    def execute(self, context: Context):
        """
        Installs or updates project dependencies.
        """
        action = context.entities.get('dependency_action', 'install')  # Default to install
        dependencies = context.entities.get('dependencies', '')
        try:
            if action == 'install':
                subprocess.run(["pip", "install"] + dependencies.split(), check=True, cwd=context.project_directory)
                print("Dependencies installed successfully.")
                logging.info("Dependencies installed successfully.")
            elif action == 'update':
                subprocess.run(["pip", "install", "--upgrade"] + dependencies.split(), check=True, cwd=context.project_directory)
                print("Dependencies updated successfully.")
                logging.info("Dependencies updated successfully.")
            else:
                print(f"Unsupported dependency action: {action}")
                logging.warning(f"Unsupported dependency action: {action}")
                return
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Dependency management failed: {e}"
            print(error_msg)
            logging.error(error_msg)


class UpdateDependenciesAction(Action):
    def execute(self, context: Context):
        """
        Updates project dependencies to their latest versions.
        """
        try:
            subprocess.run(["pip", "install", "--upgrade", "-r", "requirements.txt"], cwd=context.project_directory, check=True)
            print("Dependencies updated to latest versions.")
            logging.info("Dependencies updated to latest versions.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to update dependencies: {e}"
            print(error_msg)
            logging.error(error_msg)