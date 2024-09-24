import subprocess
from swe_actions import Action, Context

class GenerateDocsAction(Action):
    def execute(self, context: Context):
        """
        Generates documentation for the project.
        """
        doc_command = "sphinx-build -b html source/ build/"
        try:
            result = subprocess.run(doc_command, shell=True, capture_output=True, text=True, cwd=context.project_directory)
            print(f"Documentation generated successfully.")
        except Exception as e:
            print(f"Error generating documentation: {e}")
