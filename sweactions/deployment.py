import os
import logging
from swe_actions import Action, Context
import subprocess
class DockerizeApplicationAction(Action):
    def execute(self, context: Context):
        """
        Sets up Docker configurations for the application.
        """
        dockerfile_content = context.entities.get('dockerfile_content')
        if not dockerfile_content:
            dockerfile_content = self.extract_code_snippet(context.response)
        if dockerfile_content:
            dockerfile_path = os.path.join(context.project_directory, 'Dockerfile')
            try:
                with open(dockerfile_path, 'w', encoding='utf-8') as f:
                    f.write(dockerfile_content)
                print("Dockerfile created successfully.")
                logging.info("Dockerfile created successfully.")
                context.agent.update_state()
            except Exception as e:
                error_msg = f"Error creating Dockerfile: {e}"
                print(error_msg)
                logging.error(error_msg)
        else:
            error_msg = "Could not extract Dockerfile content."
            print(error_msg)
            logging.error(error_msg)

class DeployApplicationAction(Action):
    def execute(self, context: Context):
        """
        Deploys the application to a specified environment.
        """
        environment = context.entities.get('environment', 'production')
        deploy_command = context.entities.get('deploy_command', f"docker-compose up -d")
        try:
            subprocess.run(deploy_command, shell=True, check=True, cwd=context.project_directory)
            print(f"Application deployed to {environment} environment.")
            logging.info(f"Application deployed to {environment} environment.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Deployment failed: {e}"
            print(error_msg)
            logging.error(error_msg)

class MonitorLogsAction(Action):
    def execute(self, context: Context):
        """
        Monitors application logs for errors or important events.
        """
        log_command = context.entities.get('log_command', 'docker logs -f')
        try:
            subprocess.run([log_command, context.entities.get('container_name', '')], check=True, cwd=context.project_directory)
            logging.info("Started log monitoring.")
        except subprocess.CalledProcessError as e:
            error_msg = f"Log monitoring failed: {e}"
            print(error_msg)
            logging.error(error_msg)
