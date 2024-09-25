# software_agent.py
import os
import torch.nn.functional as F
import torch
import logging
import json
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from swe_actions import ActionRegistry, register_actions
from open_strawberry_torch.model import (
    TransformerPolicyNetwork as PolicyModel, 
    TransformerRewardModel as RewardModel,
    TransformerValueNetwork as ValueModel  # Import the ValueModel
)
from open_strawberry_torch.train import ThoughtTree, monte_carlo_rollout

# Define the threshold for executing actions
SOME_THRESHOLD = 0.5  # Adjust this value as appropriate

@dataclass
class Context:
    response: str
    project_directory: str
    entities: Dict[str, Any]
    # Add other shared resources as needed

class SoftwareEngineeringAgent:
    def __init__(
        self,
        project_directory: str,
        reward_model_path: Optional[str] = None,
        policy_model_path: Optional[str] = None,
        value_model_path: Optional[str] = None,  # Add path for ValueModel
        input_dim: int = 768,  # Updated to match BERT's hidden size
        action_dim: int = 10,  # Updated to match the number of actions
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        eos_token_id: int = 102,  # BERT's [SEP] token
    ):
        logging.basicConfig(
            filename='agent.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.project_directory = project_directory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.context = ""  # Conversation context
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.eos_token_id = eos_token_id

        # Initialize Policy, Reward, and Value Models
        self.policy_model = PolicyModel(
            input_dim,
            action_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        self.reward_model = RewardModel(
            input_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        self.value_model = ValueModel(  # Initialize ValueModel
            input_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self.device)
        
        if policy_model_path:
            self.policy_model.load_state_dict(torch.load(policy_model_path))
        if reward_model_path:
            self.reward_model.load_state_dict(torch.load(reward_model_path))
        if value_model_path:  # Load ValueModel weights if provided
            self.value_model.load_state_dict(torch.load(value_model_path))
        
        self.policy_model.eval()
        self.reward_model.eval()
        self.value_model.eval()  # Set ValueModel to evaluation mode
        
        # Initialize Action Registry
        self.action_registry = register_actions()
        # Map action IDs to action names (consistent with action registry)
        self.action_id_to_name: Dict[int, str] = {
            idx: name for idx, name in enumerate(self.action_registry.list_actions())
        }

        # Load pre-trained weights if available
        # self.policy_model.load_state_dict(torch.load('policy_model.pth'))
        # self.reward_model.load_state_dict(torch.load('reward_model.pth'))
        self.policy_model.eval()
        self.reward_model.eval()

        self.action_history: List[str] = []  # Initialize action history list
        self.state_file = os.path.join(self.project_directory, 'agent_state.json')
        self.load_state()  # Load existing state if available

    def get_state_representation(self) -> torch.Tensor:
        """
        Get the current state representation.

        Returns:
            torch.Tensor: Current state tensor.
        """
        # Use a representation of the current codebase
        code_texts = []
        for root, _, files in os.walk(self.project_directory):
            for file in files:
                if file.endswith('.py') or file.endswith('.txt') or file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_text = f.read()
                        code_texts.append(f"File: {file}\n{code_text}\n")
        combined_code = "\n".join(code_texts)
        
        # **Start Edit: Include Code Metrics**
        code_metrics = self.calculate_code_metrics()
        combined_code += f"\n# Code Metrics:\n{code_metrics}\n"
        # **End Edit**
        
        return combined_code

    def calculate_code_metrics(self) -> str:
        """
        Calculates various code metrics for the current project.

        Returns:
            str: A string representation of code metrics.
        """
        metrics = {}
        for root, _, files in os.walk(self.project_directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        metrics[file] = {
                            'line_count': len(lines),
                            'function_count': len([line for line in lines if line.strip().startswith('def ')]),
                            'comment_count': len([line for line in lines if line.strip().startswith('#')])
                        }
        # Convert metrics to a formatted string
        metrics_str = ""
        for file, data in metrics.items():
            metrics_str += f"File: {file}\n"
            for key, value in data.items():
                metrics_str += f"  {key}: {value}\n"
        return metrics_str

    def read_code_files(self):
        """
        Reads code files from the project directory and returns tokenized input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Token IDs and attention mask.
        """
        code_texts = []
        for root, _, files in os.walk(self.project_directory):
            for file in files:
                if file.endswith('.py') or file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_text = f.read()
                        code_texts.append(code_text)

        if code_texts:
            combined_code = "\n".join(code_texts)
            inputs = self.tokenizer(
                combined_code,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = inputs.to(self.device)
            code_input_ids = inputs['input_ids']
            code_attention_mask = inputs['attention_mask']
            return code_input_ids, code_attention_mask
        else:
            return None, None

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenizes input text.

        Args:
            text (str): Input text.

        Returns:
            torch.Tensor: Token IDs.
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=8192
        )
        return inputs['input_ids'].to(self.device)


    def detokenize(self, token_ids: torch.Tensor) -> str:
        """
        Converts token IDs back to text.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def map_response_to_actions(self, response: str) -> List[str]:
        """
        Maps the model's response text to a list of actionable action names.

        Args:
            response (str): The response text generated by the model.

        Returns:
            List[str]: A list of action names to be executed.
        """
        actions = []

        # Define a list of all possible actions to ensure accurate mapping
        possible_actions = {
            "create_file": ["create a file", "create file", "generate file"],
            "generate_content": ["generate content", "write content"],
            "add_function": ["add function", "implement function"],
            "edit_file": ["edit file", "modify file"],
            "write_tests": ["write tests", "add tests"],
            "handle_code_insertion": ["insert code", "add code"],
            "run_tests": ["run tests", "execute tests"],
            "generate_docs": ["generate docs", "create documentation"],
            "safe_execute_code": ["execute code", "run code"],
            "commit_changes": ["commit changes", "push to repository"],
            "code_review": ["perform code review", "review code", "conduct code review"],  # Added synonyms
            # Add more synonyms as needed
        }

        # Utilize regex for more flexible matching
        import re

        for action_name, keywords in possible_actions.items():
            for keyword in keywords:
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                if pattern.search(response):
                    actions.append(action_name)
                    break  # Avoid adding the same action multiple times

        # **Start Edit: Handle Exact Action Mapping from Model Output**
        # If the model outputs exact action names, map them directly
        exact_actions = [
            action.strip() for action in response.split(",") 
            if action.strip() in possible_actions
        ]
        if exact_actions:
            actions.extend(exact_actions)
        # **End Edit**

        return list(set(actions))  # Remove duplicates
    
    def execute_action(self, action_name: str, context: Context):
        """
        Executes an action based on the action name.
    
        Args:
            action_name (str): The action name to execute.
            context (Context): The context for the action.
        """
        action = self.action_registry.get_action(action_name)
        if action:
            try:
                action.execute(context.response, context)
                self.logger.info(f"Executed action: {action_name}")
                self.action_history.append(action_name)  # Track executed action
            except Exception as e:
                self.logger.error(f"Error executing action '{action_name}': {e}")
                # Optionally, notify the user or take corrective measures
                print(f"An error occurred while executing '{action_name}': {e}")
        else:
            # **Start Edit: Handle Unknown Actions Gracefully**
            self.logger.warning(f"Action not found in registry: {action_name}")
            # Optionally, notify the user or take corrective measures
            print(f"Warning: The action '{action_name}' is not recognized and cannot be executed.")
            # **End Edit**
    
    def get_user_feedback(self):
        feedback = input("Was this action helpful? (yes/no): ").strip().lower()
        return feedback == 'yes'

    def update_state(self):
        """
        Updates the agent's state after an action is executed.
        Tracks executed actions, updates context, and persists state.
        """
        # Log the current action history
        self.logger.info(f"Action History: {self.action_history}")
        
        # Update the conversation context with action history
        actions_str = ", ".join(self.action_history)
        self.context += f"Actions taken: {actions_str}\n"
        
        # **Start Edit: Incorporate User Feedback**
        feedback = self.get_user_feedback()
        self.context += f"User Feedback: {'Positive' if feedback else 'Negative'}\n"
        # **End Edit**
        
        # Persist the current state to a JSON file
        state = {
            'context': self.context,
            'action_history': self.action_history
        }
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            self.logger.info("Agent state successfully saved.")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """
        Loads the agent's state from a JSON file if it exists.
        """
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.context = state.get('context', "")
                self.action_history = state.get('action_history', [])
                self.logger.info("Agent state successfully loaded.")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
        else:
            self.logger.info("No existing state file found. Starting fresh.")