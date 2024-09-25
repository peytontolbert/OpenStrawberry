import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, BertTokenizer, BertModel
from swe_dataset import SwoDataset
from loguru import logger
from open_strawberry_torch.model import TransformerPolicyNetwork, TransformerValueNetwork, TransformerRewardModel
from open_strawberry_torch.train import ThoughtTree, monte_carlo_rollout, transition
from software_agent import SoftwareEngineeringAgent
from swe_actions import Context  # Import Context for action execution

# Set up logging
logger.add("training.log", rotation="500 MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(policy_net, value_net, reward_net, iteration, path="checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(policy_net.state_dict(), f"{path}/policy_net_{iteration}.pth")
    torch.save(value_net.state_dict(), f"{path}/value_net_{iteration}.pth")
    logger.info(f"Saved checkpoint for iteration {iteration}")

def load_checkpoint(policy_net, value_net, reward_net, iteration, path="checkpoints"):
    policy_net.load_state_dict(torch.load(f"{path}/policy_net_{iteration}.pth"))
    value_net.load_state_dict(torch.load(f"{path}/value_net_{iteration}.pth"))
    logger.info(f"Loaded checkpoint for iteration {iteration}")

def train(
    agent: SoftwareEngineeringAgent,
    policy_net: TransformerPolicyNetwork,
    value_net: TransformerValueNetwork,
    reward_model: TransformerRewardModel,
    num_iterations: int = 1000,
    episodes_per_iteration: int = 10,
    data_path: str = '',
    user_task: str = '',
    max_depth: int = 5,
    sequence_length: int = 10,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    policy_lr: float = 1e-4,
    value_lr: float = 1e-3,
    save_interval: int = 20,
    batch_size: int = 4,
    max_length: int = 8192,
    reward_shaping_factor: float = 1.0,
    reward_shaping_success: float = 0.5,
    reward_shaping_failure: float = -0.5,
):
    """
    Train the policy and value networks using PPO.

    Args:
        policy_net (TransformerPolicyNetwork): The policy network.
        value_net (TransformerValueNetwork): The value network.
        reward_model (TransformerRewardModel): The reward model.
        ... [other args]
        user_task (str): The initial task description provided by the user.
        ...
    """
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Ensure this uses BERT tokenizer
    
    train_loader, val_loader = SwoDataset.load_dataset(data_path, tokenizer, batch_size, max_length)
    
    policy_optimizer = optim.Adam(
        policy_net.parameters(), lr=policy_lr
    )
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    best_validation_score = float('-inf')
    early_stopping_counter = 0
    early_stopping_patience = 10  # Number of iterations to wait before stopping
    
    for iteration in range(num_iterations):
        logger.info(
            f"Starting iteration {iteration + 1}/{num_iterations}"
        )
        memory = []

        for episode in range(episodes_per_iteration):
            logger.debug(
                f"Starting episode {episode + 1}/{episodes_per_iteration}"
            )
            # Initialize state_sequence with user_task and tokenized code files
            # Encode the user task
            task_tokens = tokenizer.encode(user_task, return_tensors='pt').to(device)
            
            # Get tokenized code files from the agent's state representation
            code_representation = agent.get_state_representation()
            code_tokens = tokenizer.encode(code_representation, return_tensors='pt').to(device)
            
            # Combine task and code tokens
            initial_state = torch.cat((task_tokens, code_tokens), dim=1)
            state_sequence = initial_state
            # Ensure the sequence does not exceed the maximum length
            if state_sequence.size(1) > sequence_length:
                state_sequence = state_sequence[:, -sequence_length:]

            thought_tree = ThoughtTree(state_sequence)
            trajectory = []
            # Generate thought branches
            for depth in range(max_depth):
                # Expand dimensions to match (sequence_length, batch_size, input_dim)
                src = state_sequence.unsqueeze(
                    1
                )  # Shape: (sequence_length, 1, input_dim)
                action_logits = policy_net(src)  # Get logits instead of probabilities
                m = Categorical(action_logits)
                actions = m.sample((5,))  # Generate multiple branches
                rewards = []
                log_probs = []  # Initialize list to store log_probs

                for action in actions:
                    next_state = transition(
                        state_sequence[:, -1], action
                    )
                    # Update the sequence by appending the new state
                    next_sequence = torch.cat(
                        [state_sequence, next_state.unsqueeze(0)],
                        dim=1,
                    )
                    # Ensure the sequence length does not exceed the maximum
                    if next_sequence.size(1) > sequence_length:
                        next_sequence = next_sequence[:, -sequence_length:]
                    rollout = monte_carlo_rollout(
                        policy_net,
                        next_sequence,
                        depth + 1,
                        max_depth,
                        sequence_length,
                    )
                    total_reward = sum([r for _, r in rollout])
                    # Expand dimensions for reward model input
                    reward_input = next_sequence.unsqueeze(1)
                    reward_estimate = reward_model(reward_input)
                    reward = reward_estimate.item() + total_reward
                    rewards.append(reward)

                    # Update thought tree
                    thought_tree.add_child(
                        thought_tree.root, next_sequence, reward
                    )

                    # Store the log probability of the action
                    log_prob = m.log_prob(action)
                    log_probs.append(log_prob.detach())

                # Select the best action based on rewards
                best_action_index = (
                    torch.tensor(rewards).argmax().item()
                )
                best_action = actions[best_action_index]
                best_reward = rewards[best_action_index]
                best_log_prob = log_probs[best_action_index]  # Retrieve stored log_prob

                # Convert Logits to Actions Using Tokenizer and Parsing Function
                # Decode the best action using the tokenizer
                action_id = best_action.item()
                action_text = tokenizer.decode([action_id], skip_special_tokens=True)
                
                # Map the action text to actionable commands
                action_names = agent.map_response_to_actions(action_text)
                
                for action_name in action_names:
                    # Create a Context object for the action
                    context = Context(
                        response="Training iteration action execution",
                        project_directory=agent.project_directory,
                        entities={}  # Populate with relevant entities if needed
                    )
                    # Execute the parsed action
                    agent.execute_action(action_name, context)
                    
                    # Update state_sequence with the latest state from the agent after action execution
                    # Fetch the updated state representation from the agent
                    updated_state = agent.get_state_representation()
                    
                    # Tokenize the updated state
                    updated_state_tokens = tokenizer.encode(updated_state, return_tensors='pt').to(device)
                    
                    # Combine the existing state_sequence with the updated state
                    state_sequence = torch.cat([state_sequence, updated_state_tokens], dim=1)
                    
                    # Ensure the state_sequence does not exceed the maximum length
                    if state_sequence.size(1) > sequence_length:
                        state_sequence = state_sequence[:, -sequence_length:]
                
                # Store the experience with old_log_prob
                trajectory.append(
                    (state_sequence.clone(), best_action, best_reward, best_log_prob)  # Updated reward
                )

            # Compute returns and advantages with shaped rewards
            returns = []
            advantages = []
            Gt = 0
            for state_seq_t, action_t, reward_t, _ in reversed(
                trajectory
            ):
                state_value = value_net(state_seq_t.unsqueeze(1)).item()
                Gt = reward_shaping_factor * reward_t + gamma * Gt + gamma * state_value
                returns.insert(0, Gt)
                advantage = Gt - state_value
                advantages.insert(0, advantage)

            advantages_tensor = torch.tensor(
                advantages, dtype=torch.float32
            ).to(device)
            if advantages_tensor.std() != 0:
                advantages_tensor = (
                    advantages_tensor - advantages_tensor.mean()
                ) / (advantages_tensor.std() + 1e-8)
            else:
                advantages_tensor = advantages_tensor - advantages_tensor.mean()
        
            # Update policy network using PPO
            policy_losses = []  # Collect policy losses for averaging
            for i, (state_seq_t, action_t, _, old_log_prob) in enumerate(
                trajectory
            ):
                # Expand dimensions to match (sequence_length, batch_size, input_dim)
                src = state_seq_t.unsqueeze(1)
                action_logits = policy_net(src)
                m = Categorical(action_logits)
                log_prob = m.log_prob(action_t)
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantages_tensor[i]
                surr2 = (
                    torch.clamp(
                        ratio, 1 - clip_epsilon, 1 + clip_epsilon
                    )
                    * advantages_tensor[i]
                )
                policy_loss = -torch.min(surr1, surr2)
                policy_losses.append(policy_loss)

                # Log the policy loss
                logger.debug(
                    f"Policy loss at step {i}: {policy_loss.item()}"
                )
        
            # Aggregate policy loss and perform backpropagation
            if policy_losses:
                total_policy_loss = torch.stack(policy_losses).mean()
                policy_optimizer.zero_grad()
                total_policy_loss.backward()
                policy_optimizer.step()

            # Update value network
            returns_tensor = (
                torch.tensor(returns, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            # Prepare inputs for the value network
            value_inputs = torch.stack(
                [s for s, _, _, _ in trajectory]
            ).transpose(0, 1)
            value_inputs = value_inputs.to(device)
            values = value_net(value_inputs)
            value_loss = nn.MSELoss()(values, returns_tensor)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # Log the value loss
            logger.debug(f"Value loss: {value_loss.item()}")

        
        if (iteration + 1) % save_interval == 0:
            save_checkpoint(policy_net, value_net, reward_model, iteration + 1)  # Pass reward_model
        logger.info(
            f"Completed iteration {iteration + 1}/{num_iterations}"
        )



