from swe_actions import Action, Context
from chat_with_ollama import ChatGPT
import logging
import os
class GenerateContentAction(Action):
    def execute(self, context: Context):
        """
        Generates content based on the provided instructions and updates the agent's state.
        """
        content_instructions = context.entities.get('content_instructions', context.response)
        if content_instructions:
            # Implement content generation logic here (e.g., using a language model)
            generated_content = self.generate_content(content_instructions)
            # Assume there's a target file or location to insert the generated content
            target_file = context.entities.get('target_file', 'README.md')
            file_path = os.path.join(context.project_directory, target_file)
            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write("\n" + generated_content)
                print(f"Generated content in {target_file}")
                logging.info(f"Generated content in {target_file}")
                context.agent.update_state()
            except Exception as e:
                error_msg = f"Error generating content in {target_file}: {e}"
                print(error_msg)
                logging.error(error_msg)
        else:
            error_msg = "No content instructions provided."
            print(error_msg)
            logging.error(error_msg)
    
    def generate_content(self, instructions: str) -> str:
        """
        Placeholder method for content generation. Integrate with actual content generation logic.
    
        Args:
            instructions (str): Instructions for content generation.
    
        Returns:
            str: Generated content.
        """
        # Placeholder implementation
        return f"# Generated Section\n\n{instructions}"


class CodeReviewAction(Action):
    def execute(self, context: Context):
        """
        Performs a code review and provides feedback.
        """
        # Placeholder for actual code review logic, possibly integrating with a code review tool or LLM
        feedback = ChatGPT.chat_with_ollama(f"Review the following code:\n{context.entities.get('code_snippet', '')}")
        print(f"Code Review Feedback:\n{feedback}")
        logging.info("Performed code review.")
        context.agent.update_state()
