# eval.py
import os
import torch
import json
import logging
from typing import List, Dict, Optional

from software_agent import SoftwareEngineeringAgent
from open_strawberry_torch.model import TransformerPolicyNetwork as PolicyModel, TransformerRewardModel as RewardModel

# Configure logging
logging.basicConfig(
    filename='evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_models(
    policy_model_path: str,
    reward_model_path: str,
    device: torch.device
) -> Dict[str, torch.nn.Module]:
    """
    Load the trained policy and reward models.

    Args:
        policy_model_path (str): Path to the trained policy model.
        reward_model_path (str): Path to the trained reward model.
        device (torch.device): Device to load the models onto.

    Returns:
        Dict[str, torch.nn.Module]: Dictionary containing loaded models.
    """
    policy_model = PolicyModel().to(device)
    reward_model = RewardModel().to(device)

    if os.path.exists(policy_model_path):
        policy_model.load_state_dict(torch.load(policy_model_path, map_location=device))
        logger.info(f"Loaded policy model from {policy_model_path}")
    else:
        logger.error(f"Policy model file not found at {policy_model_path}")
        raise FileNotFoundError(f"Policy model file not found at {policy_model_path}")

    if os.path.exists(reward_model_path):
        reward_model.load_state_dict(torch.load(reward_model_path, map_location=device))
        logger.info(f"Loaded reward model from {reward_model_path}")
    else:
        logger.error(f"Reward model file not found at {reward_model_path}")
        raise FileNotFoundError(f"Reward model file not found at {reward_model_path}")

    policy_model.eval()
    reward_model.eval()

    return {
        'policy_model': policy_model,
        'reward_model': reward_model
    }

def evaluate_agent(agent: SoftwareEngineeringAgent, test_inputs: List[str]) -> Dict[str, float]:
    """
    Evaluate the agent on a set of test inputs.

    Args:
        agent (SoftwareEngineeringAgent): The agent to evaluate.
        test_inputs (List[str]): A list of user input strings for testing.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics.
    """
    total_actions = 0
    successful_actions = 0
    total_responses = 0
    successful_responses = 0

    for user_input in test_inputs:
        total_responses += 1
        try:
            agent.process_user_input(user_input)
            # Assuming execute_action updates successful_actions
            # This logic may vary based on implementation
            if agent.action_history:
                successful_actions += len(agent.action_history)
            successful_responses += 1
            logger.info(f"Processed input: {user_input}")
        except Exception as e:
            logger.error(f"Error processing input '{user_input}': {e}")

        # Reset action history for next input
        agent.action_history = []

    metrics = {
        'Total Inputs': len(test_inputs),
        'Successful Responses': successful_responses,
        'Success Rate (%)': (successful_responses / len(test_inputs)) * 100 if test_inputs else 0,
        'Total Actions Executed': successful_actions,
        'Average Actions per Input': (successful_actions / len(test_inputs)) if test_inputs else 0
    }

    return metrics

def main():
    # Paths to the trained models
    policy_model_path = 'trained_policy_model.pth' 
    reward_model_path = 'trained_reward_model.pth' 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained models
    models = load_models(policy_model_path, reward_model_path, device)

    # Initialize the agent with loaded models
    agent = SoftwareEngineeringAgent(
        project_directory="virtual_env",
        policy_model=models['policy_model'],
        reward_model=models['reward_model'],
        # Include other necessary parameters if any
    )

    # Define test inputs for evaluation
    test_inputs = [
        "Create a basic hello world python script.",
        "Generate a simple python script that calculates the sum of two numbers.",
        "Generate documentation for the API endpoints."
        "Create a simple python script that calculates the factorial of a number."
        
        # Add more test cases as needed
    ]

    # Run evaluation
    metrics = evaluate_agent(agent, test_inputs)

    # Log and print evaluation metrics
    logger.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
