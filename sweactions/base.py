import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import re

@dataclass
class Context:
    entities: Dict[str, Any]
    project_directory: str

class Action(ABC):
    @abstractmethod
    def execute(self, context: Context):
        pass

    @staticmethod
    def extract_file_name(text: str) -> Optional[str]:
        match = re.search(r'file named (\w+\.\w+)', text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_function_name(text: str) -> Optional[str]:
        match = re.search(r'function called (\w+)', text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_code_snippet(text: str) -> Optional[str]:
        match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_file_path(text: str) -> Optional[str]:
        match = re.search(r'in file ([\w./]+)', text)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def extract_edit_instructions(text: str) -> str:
        """
        Extracts edit instructions from the text.
        """
        # For simplicity, we'll assume the entire text is the edit instruction
        return text
