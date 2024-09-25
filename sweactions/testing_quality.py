from swe_actions import Action, Context
import os
import logging
import subprocess

class WriteTestsAction(Action):
    def execute(self, context: Context):
        """
        Writes tests for specified functions or modules and updates the agent's state.
        """
        file_name = context.entities.get('file_name', 'test.py')
        test_code = context.entities.get('test_code')
        
        if not test_code:
            test_code = self.extract_code_snippet(context.response)
    
        if test_code:
            file_path = os.path.join(context.project_directory, file_name)
            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write("\n" + test_code)
                print(f"Added tests to {file_name}")
                logging.info(f"Added tests to {file_name}")
                context.agent.update_state()  # Update the agent's state after action
            except Exception as e:
                error_msg = f"Error writing tests to {file_name}: {e}"
                print(error_msg)
                logging.error(error_msg)
        else:
            error_msg = "Could not extract test code."
            print(error_msg)
            logging.error(error_msg)

class RunTestsAction(Action):
    def execute(self, context: Context):
        """
        Runs the test suite for the project.
        """
        test_command = "python -m unittest"
        try:
            result = subprocess.run(test_command, shell=True, capture_output=True, text=True, cwd=context.project_directory)
            print(f"Test Results:\n{result.stdout}")
        except Exception as e:
            print(f"Error running tests: {e}")

class AnalyzeTestCoverageAction(Action):
    def execute(self, context: Context):
        """
        Checks the test coverage of the project using tools like coverage.py.
        """
        coverage_command = context.entities.get('coverage_command', 'coverage run -m unittest discover && coverage report')
        try:
            result = subprocess.run(coverage_command, shell=True, capture_output=True, text=True, check=True, cwd=context.project_directory)
            print(f"Test Coverage Report:\n{result.stdout}")
            logging.info("Test coverage analysis completed.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Test coverage analysis failed: {e}\n{e.stdout}\n{e.stderr}"
            print(error_msg)
            logging.error(error_msg)

class AnalyzeCodeQualityAction(Action):
    def execute(self, context: Context):
        """
        Analyzes code quality using tools like pylint or flake8.
        """
        tool = context.entities.get('quality_tool', 'pylint')  # Default to pylint
        try:
            if tool == 'pylint':
                command = ["pylint", context.project_directory]
            elif tool == 'flake8':
                command = ["flake8", context.project_directory]
            else:
                print(f"Unsupported code quality tool: {tool}")
                logging.warning(f"Unsupported code quality tool: {tool}")
                return

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"Code Quality Report ({tool}):\n{result.stdout}")
            logging.info(f"Code quality analysis using {tool} completed.")
            context.agent.update_state()
        except subprocess.CalledProcessError as e:
            error_msg = f"Code quality analysis failed: {e}\n{e.stdout}\n{e.stderr}"
            print(error_msg)
            logging.error(error_msg)

