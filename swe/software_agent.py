# software_agent.py
import os
import torch.nn.functional as F
import torch
import logging
import json
from typing import List, Dict, Callable, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from swe_actions import ActionRegistry, register_actions, Context
from open_strawberry_torch.model import (
    TransformerPolicyNetwork as PolicyModel, 
    TransformerRewardModel as RewardModel,
    TransformerValueNetwork as ValueModel  # Import the ValueModel
)
from open_strawberry_torch.train import ThoughtTree, monte_carlo_rollout

# Define the threshold for executing actions
SOME_THRESHOLD = 0.5  # Adjust this value as appropriate

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
        return combined_code



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
        return inputs.input_ids.to(self.device)


    def detokenize(self, token_ids: torch.Tensor) -> str:
        """
        Converts token IDs back to text.

        Args:
            token_ids (torch.Tensor): Tensor of token IDs.

        Returns:
            str: Decoded text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def process_user_input(self, user_input: str):
        """
        Processes user input and updates the agent's state.

        Args:
            user_input (str): The user input.
        """
        # Update conversation context
        self.context += f"User: {user_input}\n"
        # Get current state representation
        state_representation = self.get_state_representation()
        prompt = self.context + state_representation + f"\nAgent:"
        # Generate a response from the policy model
        response = self.generate_response(prompt)
        # Append the agent's response to the conversation context
        self.context += f"{response}\n"
        print(f"Agent: {response}")

        # Map response to actions
        actions = self.map_response_to_actions(response)

        # **Start Edit: Handle Multiple Actions**
        if isinstance(actions, list):
            for action_name in actions:
                execution_context = Context(
                    response=response,
                    project_directory=self.project_directory,
                    agent=self,
                    entities={}  # You can extract entities as needed
                )
                self.execute_action(action_name, execution_context)
        else:
            execution_context = Context(
                response=response,
                project_directory=self.project_directory,
                entities={}  # You can extract entities as needed
            )
            self.execute_action(actions, execution_context)
        # **End Edit**

        # Update the agent's state if necessary
        self.update_state()

    def map_response_to_actions(self, response: str) -> List[str]:
        """
        Parses the agent's response to extract action commands.

        Args:
            response (str): The agent's generated response.

        Returns:
            List[str]: A list of action names to execute.
        """
        actions = []

        # Example parsing logic: look for action keywords in the response
        action_keywords = {
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
        }

        for action_name, keywords in action_keywords.items():
            for keyword in keywords:
                if keyword in response.lower():
                    actions.append(action_name)
                    break  # Avoid adding the same action multiple times

        return actions
    
    def execute_action(self, action_name: str, context: Context):
        """
        Executes an action based on the action name.
    
        Args:
            action_name (str): The action name to execute.
            context (Context): The context for the action.
        """
        action = self.action_registry.get_action(action_name)
        if action:
            action.execute(context.response, context)
            self.logger.info(f"Executed action: {action_name}")
            self.action_history.append(action_name)  # Track executed action
            
            # Example Usage of ValueModel
            state_representation = self.get_state_representation()
            state_tensor = self.tokenize(state_representation)
            state_value = self.value_model(state_tensor)
            self.logger.info(f"State Value after action '{action_name}': {state_value.item()}")
        else:
            self.logger.warning(f"Action not found in registry: {action_name}")
    
    def sample_sequence(
        self,
        context: List[int],
        max_length: int = 50
    ) -> List[int]:
        """
        Samples a continuation from the policy model given the context.

        Args:
            context (List[int]): The context sequence (list of token IDs).
            max_length (int): Maximum length of the continuation.

        Returns:
            List[int]: Sampled continuation tokens.
        """
        generated = context.copy()
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = torch.tensor([generated]).to(self.device)
                logits = self.policy_model(input_ids)
                next_token_logits = logits[0, -1, :]
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()
                generated.append(next_token_id)
                if next_token_id == self.eos_token_id:
                    break
        continuation = generated[len(context):]
        return continuation

    def compute_reward(self, sequence: List[int]) -> float:
        """
        Computes the reward of a sequence using the reward model.

        Args:
            sequence (List[int]): The sequence of token IDs.

        Returns:
            float: Reward value.
        """
        with torch.no_grad():
            input_ids = torch.tensor([sequence]).to(self.device)
            reward = self.reward_model(input_ids)
            return reward.item()

    def get_user_feedback(self):
        feedback = input("Was this action helpful? (yes/no): ").strip().lower()
        return feedback == 'yes'

    def log_unsuccessful_interaction(self, user_input, response):
        with open('unsuccessful_interactions.log', 'a', encoding='utf-8') as f:
            f.write(f"User Input: {user_input}\n")
            f.write(f"Agent Response: {response}\n")
            f.write("-----\n")
        self.logger.info("Logged unsuccessful interaction for future training.")

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