import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer
from swe_dataset import SwoDataset
from loguru import logger
from open_strawberry_torch.model import TransformerPolicyNetwork, TransformerValueNetwork, TransformerRewardModel
from open_strawberry_torch.train import ThoughtTree, monte_carlo_rollout, transition
from software_agent import SoftwareEngineeringAgent

# Set up logging
logger.add("training.log", rotation="500 MB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(policy_net, value_net, iteration, path="checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(policy_net.state_dict(), f"{path}/policy_net_{iteration}.pth")
    torch.save(value_net.state_dict(), f"{path}/value_net_{iteration}.pth")
    logger.info(f"Saved checkpoint for iteration {iteration}")

def load_checkpoint(policy_net, value_net, iteration, path="checkpoints"):
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
    tokenizer_name: str = 'bert-base-uncased',
    max_depth: int = 5,
    sequence_length: int = 10,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    policy_lr: float = 1e-4,
    value_lr: float = 1e-3,
    save_interval: int = 20,
    batch_size: int = 4,
    max_length: int = 8192,
    reward_shaping_factor: float = 1.0,         # Added definition
    reward_shaping_success: float = 0.5,       # Added definition
    reward_shaping_failure: float = -0.5,      # Added definition
):
    """
    Train the policy and value networks using PPO.

    Args:
        policy_net (TransformerPolicyNetwork): The policy network.
        value_net (TransformerValueNetwork): The value network.
        reward_model (TransformerRewardModel): The reward model.
        num_iterations (int): Number of training iterations.
        episodes_per_iteration (int): Episodes per iteration.
        max_depth (int): Maximum depth for Monte Carlo rollouts.
        sequence_length (int): Maximum sequence length for the transformer.
        gamma (float): Discount factor.
        clip_epsilon (float): Clipping epsilon for PPO.
        policy_lr (float): Learning rate for the policy optimizer.
        value_lr (float): Learning rate for the value optimizer.
        reward_shaping_factor (float): Factor to scale the shaped rewards.
        reward_shaping_success (float): Reward adjustment for successful feedback.
        reward_shaping_failure (float): Reward adjustment for failed feedback.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load datasets
    train_loader, val_loader = SwoDataset.load_dataset(data_path, tokenizer, batch_size, max_length)
    
    policy_optimizer = optim.Adam(
        policy_net.parameters(), lr=policy_lr
    )
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    for iteration in range(num_iterations):
        logger.info(
            f"Starting iteration {iteration + 1}/{num_iterations}"
        )
        memory = []

        for episode in range(episodes_per_iteration):
            logger.debug(
                f"Starting episode {episode + 1}/{episodes_per_iteration}"
            )
            # Initialize state sequence with zeros
            state = torch.zeros(policy_net.embedding.in_features).to(
                device
            )
            state_sequence = state.unsqueeze(
                0
            )  # Shape: (1, input_dim)
            thought_tree = ThoughtTree(state_sequence)
            trajectory = []

            # Generate thought branches
            for depth in range(max_depth):
                # Expand dimensions to match (sequence_length, batch_size, input_dim)
                src = state_sequence.unsqueeze(
                    1
                )  # Shape: (sequence_length, 1, input_dim)
                action_probs = policy_net(src)
                m = Categorical(action_probs)
                actions = m.sample((5,))  # Generate multiple branches
                rewards = []

                for action in actions:
                    next_state = transition(
                        state_sequence[-1], action
                    )
                    # Update the sequence by appending the new state
                    next_sequence = torch.cat(
                        [state_sequence, next_state.unsqueeze(0)],
                        dim=0,
                    )
                    # Ensure the sequence length does not exceed the maximum
                    if next_sequence.size(0) > sequence_length:
                        next_sequence = next_sequence[1:, :]
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

                # Select the best action based on rewards
                best_action_index = (
                    torch.tensor(rewards).argmax().item()
                )
                best_action = actions[best_action_index]
                best_reward = rewards[best_action_index]

                # **Start Edit: Remove Feedback Mechanism and Use Reward Model**
                # Compute adjusted reward using the reward model
                adjusted_reward = reward_model(state_sequence).item()
                rewards[best_action_index] = adjusted_reward
                # **End Edit**

                # Log the selected action and reward
                logger.debug(
                    f"Selected action {best_action.item()} with reward {best_reward}"
                )

                # Store the experience
                trajectory.append(
                    (state_sequence.clone(), best_action, adjusted_reward)  # Updated reward
                )

                # Move to the next state sequence
                next_state = transition(
                    state_sequence[-1], best_action
                )
                state_sequence = torch.cat(
                    [state_sequence, next_state.unsqueeze(0)], dim=0
                )
                if state_sequence.size(0) > sequence_length:
                    state_sequence = state_sequence[1:, :]

            # Compute returns and advantages with shaped rewards
            returns = []
            advantages = []
            Gt = 0
            for state_seq_t, action_t, reward_t in reversed(
                trajectory
            ):
                Gt = reward_shaping_factor * reward_t + gamma * Gt  # Modified reward calculation
                returns.insert(0, Gt)
                # Expand dimensions for value network input
                value_input = state_seq_t.unsqueeze(1)
                state_value = value_net(value_input)
                advantage = Gt - state_value.item()
                advantages.insert(0, advantage)

            # Normalize advantages
            advantages_tensor = torch.tensor(
                advantages, dtype=torch.float32
            ).to(device)
            advantages_tensor = (
                advantages_tensor - advantages_tensor.mean()
            ) / (advantages_tensor.std() + 1e-8)

            # Update policy network using PPO
            for i, (state_seq_t, action_t, _) in enumerate(
                trajectory
            ):
                # Expand dimensions to match (sequence_length, batch_size, input_dim)
                src = state_seq_t.unsqueeze(1)
                action_probs = policy_net(src)
                m = Categorical(action_probs)
                log_prob = m.log_prob(action_t)
                old_log_prob = log_prob.detach()
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantages_tensor[i]
                surr2 = (
                    torch.clamp(
                        ratio, 1 - clip_epsilon, 1 + clip_epsilon
                    )
                    * advantages_tensor[i]
                )
                policy_loss = -torch.min(surr1, surr2)

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Log the policy loss
                logger.debug(
                    f"Policy loss at step {i}: {policy_loss.item()}"
                )

            # Update value network
            returns_tensor = (
                torch.tensor(returns, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            # Prepare inputs for the value network
            value_inputs = torch.stack(
                [s for s, _, _ in trajectory]
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
            save_checkpoint(policy_net, value_net, iteration + 1)

        logger.info(
            f"Completed iteration {iteration + 1}/{num_iterations}"
        )



