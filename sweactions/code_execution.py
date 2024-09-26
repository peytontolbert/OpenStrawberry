import os
from swe.swe_actions import Action, Context


class SafeExecuteCodeAction(Action):
    def execute(self, context: Context):
        """
        Safely executes the python file_path after performing safety checks.
        """
        code_snippet = self.extract_code_snippet(context.response)
        file_path = self.extract_file_path(context.response)
        if code_snippet and file_path:
            full_path = os.path.join(context.project_directory, file_path)
            try:
                # Read the content of the file
                with open(full_path, 'r') as file:
                    code = file.read()
                
                # Perform safety checks (you may want to implement more thorough checks)
                if 'import os' in code or 'import sys' in code:
                    raise ValueError("Potentially unsafe operations detected")
                
                # Execute the code in a restricted environment
                exec(code, {'__builtins__': {}})
                print(f"Executed code in {file_path}")
            except Exception as e:
                print(f"Error executing code in {file_path}: {e}")
        else:
            print("Could not extract code snippet or file path.")
   