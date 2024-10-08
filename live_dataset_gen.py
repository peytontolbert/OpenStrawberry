from transformers import pipeline
import torch
import json
from typing import List, Dict
from loguru import logger

# Set up logging
logger.add("dataset.log", rotation="500 MB")

class LiveDatasetGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            device=0 if torch.cuda.is_available() else -1
        )  # {{ edit_2 }}
    
    def generate_example(self, prompt: str) -> dict:
        response = self.generator(
            prompt,
            max_length=500,  # Increased length to better accommodate all fields  # {{ edit_5 }}
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
        generated_text = response[0]['generated_text'].strip()
        logger.debug(f"Generated text: {generated_text}")  # {{ new debug }}
        # Parse the generated text into the required fields
        example = self.parse_generated_text(generated_text)
        return example  # {{ edit_6 }}
    
    def parse_generated_text(self, text: str) -> dict:
        """
        Parses the generated text to extract structured data.
        
        Args:
            text (str): The raw text generated by the language model.
        
        Returns:
            dict: A dictionary containing the structured task data.
        """
        try:
            # Attempt to parse the text as JSON
            example = json.loads(text)
        except json.JSONDecodeError:
            # If parsing fails, return a default structure with empty fields
            example = {
                "task_id": f"task_{int(torch.randint(0, 1000000, (1,)).item())}",
                "description": text,
                "code_files": [],  # {{ edit_7 }}
                "previous_actions": [],
                "chosen_action": {},
                "future_actions": []
            }
            logger.warning("Failed to parse JSON. Returning default example with empty code_files.")  # {{ edit_8 }}
        return example
    
    def get_batch(self, batch_size: int, max_attempts_per_example: int = 5) -> List[Dict]:
        """
        Generates a batch of valid training examples.
        
        Args:
            batch_size (int): Number of valid examples to generate.
            max_attempts_per_example (int): Maximum attempts to generate a valid example per batch item.
        
        Returns:
            List[Dict]: A list of valid training examples.
        """
        batch = []
        for _ in range(batch_size):
            attempts = 0
            while attempts < max_attempts_per_example:
                prompt = self._create_prompt()
                example = self.generate_example(prompt)
                if self.validate_example(example):
                    batch.append(example)
                    break
                else:
                    attempts += 1
                    logger.warning(f"Attempt {attempts} for example generation failed. Retrying...")  # {{ edit_8 }}
            if attempts == max_attempts_per_example:
                logger.error("Max attempts reached. Could not generate a valid example.")
        return batch
    
    def _create_prompt(self) -> str:
        # Define how prompts are created for the LLM to encourage structured JSON output
        return (
            "Generate a comprehensive software engineering task example in JSON format for training purposes. "
            "Include the following fields: "
            "task_id, description, code_files (each with file_path and content), "
            "previous_actions (each with action_id, action_type, result), "
            "chosen_action (with action_id, action_type, details), "
            "future_actions (each with action_id, action_type, details). "
            "Ensure the JSON is properly formatted and that 'code_files' contains at least one file with valid 'file_path' and 'content'.\n\n"
            "Example:\n"
            "{\n"
            '  "task_id": "task_1",\n'
            '  "description": "Implement user authentication module.",\n'
            '  "code_files": [\n'
            '    {\n'
            '      "file_path": "auth.py",\n'
            '      "content": "# Authentication logic here"\n'
            '    }\n'
            '  ],\n'
            '  "previous_actions": [],\n'
            '  "chosen_action": {},\n'
            '  "future_actions": []\n'
            "}"
        )  # {{ edit_4 }}
    
    def validate_example(self, example: dict) -> bool:
        """
        Validates that the generated example contains all required fields and that 'code_files' is not empty.
        
        Args:
            example (dict): The generated task example.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        required_fields = {
            "task_id",
            "description",
            "code_files",
            "previous_actions",
            "chosen_action",
            "future_actions"
        }
        has_required_fields = required_fields.issubset(example.keys())
        has_code_files = bool(example.get("code_files"))
        are_code_files_valid = all(
            isinstance(cf, dict) and 'file_path' in cf and 'content' in cf for cf in example.get("code_files", [])
        )
        if not has_code_files:
            logger.warning("Generated example has empty 'code_files'.")  # {{ edit_9 }}
        if has_code_files and not are_code_files_valid:
            logger.warning("One or more 'code_files' entries are invalid.")  # {{ new warning }}
        return has_required_fields and has_code_files and are_code_files_valid
    