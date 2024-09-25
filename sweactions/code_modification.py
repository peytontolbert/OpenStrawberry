from swe_actions import Action, Context
import os
import logging
from chat_with_ollama import ChatGPT
import subprocess
class RefactorCodeAction(Action):
    def execute(self, context: Context):
        """
        Refactors existing code for optimization and readability.
        """
        code_snippet = context.entities.get('code_snippet')
        if not code_snippet:
            code_snippet = self.extract_code_snippet(context.response)
        if code_snippet:
            refactored_code = ChatGPT.chat_with_ollama(f"Refactor the following code for better readability and performance:\n{code_snippet}")
            file_path = self.extract_file_path(context.response)
            if file_path:
                full_path = os.path.join(context.project_directory, file_path)
                try:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(refactored_code)
                    print(f"Refactored code in {file_path}")
                    logging.info(f"Refactored code in {file_path}")
                    context.agent.update_state()
                except Exception as e:
                    error_msg = f"Error refactoring code in {file_path}: {e}"
                    print(error_msg)
                    logging.error(error_msg)
            else:
                error_msg = "Could not extract file path for refactoring."
                print(error_msg)
                logging.error(error_msg)
        else:
            error_msg = "Could not extract code snippet for refactoring."
            print(error_msg)
            logging.error(error_msg)

class FormatCodeAction(Action):
    def execute(self, context: Context):
        """
        Formats code using tools like black or yapf.
        """
        formatter = context.entities.get('formatter', 'black')  # Default to black
        try:
            if formatter == 'black':
                command = ["black", context.project_directory]
            elif formatter == 'yapf':
                command = ["yapf", "-i", "-r", context.project_directory]
            else:
                print(f"Unsupported formatter: {formatter}")
                logging.warning(f"Unsupported formatter: {formatter}")
                return

            subprocess.run(command, check=True)
            print(f"Code formatted using {formatter}.")
            logging.info(f"Code formatted using {formatter}.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Code formatting failed: {e}"
            print(error_msg)
            logging.error(error_msg)
