import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from transformers import AutoTokenizer, BertTokenizer, BertModel, pipeline
from swe.swe_dataset import SwoDataset
from loguru import logger
from open_strawberry_torch.model import TransformerPolicyNetwork, TransformerValueNetwork, TransformerRewardModel
from open_strawberry_torch.train import ThoughtTree, monte_carlo_rollout, transition
from swe.software_agent import SoftwareEngineeringAgent
from swe.swe_actions import Context  # Import Context for action execution
from open_strawberry_torch.dpo import DPO  # Imported DPO and loss computation
from live_dataset_gen import LiveDatasetGenerator
import torch.nn.functional as F  # {{ edit_48 }}
import random 
import numpy as np 

# Set up logging
logger.add("training.log", rotation="500 MB")

# {{ edit_53 }} Set random seeds for reproducibility
def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds set to {seed}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seeds()

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

def initialize_prompt_mask(initial_state, tokenizer):
    """
    Initializes the prompt_mask tensor to distinguish between prompt and response tokens.
    
    Args:
        initial_state (torch.Tensor): Tensor containing the initial state sequence.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for encoding.
    
    Returns:
        prompt_mask (torch.Tensor): Tensor mask with 1 for prompt tokens and 0 for response tokens.
    """
    prompt_length = len(tokenizer.decode(initial_state[0], skip_special_tokens=True).split())
    mask = torch.ones(initial_state.size(1), dtype=torch.bool).to(device)
    mask[:prompt_length] = 1
    mask[prompt_length:] = 0
    return mask

def determine_preferences(generated_sequences, preference_pipeline):  # {{ edit_49 }}
    """
    Determines the preferred and unpreferred sequences based on sentiment analysis.
    
    Args:
        generated_sequences (List[str]): List of generated sequence strings.
        preference_pipeline (transformers.Pipeline): Sentiment analysis pipeline.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: Preferred and unpreferred sequences.
    """
    sentiments = preference_pipeline(generated_sequences)
    preferred_seq = None
    unpreferred_seq = None
    for seq, sentiment in zip(generated_sequences, sentiments):
        if sentiment['label'] == 'POSITIVE' and (preferred_seq is None or sentiment['score'] > preference_pipeline(preferred_seq)[0]['score']):
            preferred_seq = seq
        elif sentiment['label'] == 'NEGATIVE' and (unpreferred_seq is None or sentiment['score'] > preference_pipeline(unpreferred_seq)[0]['score']):
            unpreferred_seq = seq

    # {{ edit_50 }} Handle cases where preferred or unpreferred sequences might not be found
    if not preferred_seq:
        logger.warning("No preferred sequence found based on sentiment analysis.")
    if not unpreferred_seq:
        logger.warning("No unpreferred sequence found based on sentiment analysis.")

    return preferred_seq, unpreferred_seq

def train(
    agent: SoftwareEngineeringAgent,
    policy_net: TransformerPolicyNetwork,
    value_net: TransformerValueNetwork,  # Retained value network
    reward_model: TransformerRewardModel,
    num_iterations: int = 1000,
    input_dim: int = 768,
    action_dim: int = 10,
    episodes_per_iteration: int = 10,
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
    model_name: str = 'gpt2',  # {{ edit_4 }}
):
    """
    Train the policy and value networks using PPO and DPO with Iteration of Thoughts.
    
    Args:
        policy_net (TransformerPolicyNetwork): The policy network.
        value_net (TransformerValueNetwork): The value network.
        reward_model (TransformerRewardModel): The reward model.
        ... [other args]
        user_task (str): The initial task description provided by the user.
        ...
    """
    logger.debug("Starting training with num_iterations={}, episodes_per_iteration={}".format(num_iterations, episodes_per_iteration))  # {{ edit_1 }}
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, max_length=max_length)  # Ensure this uses BERT tokenizer and handle truncation
    logger.debug("BERT tokenizer initialized")  # {{ edit_2 }}
    
    # Initialize DPO
    dpo_model = DPO(model=policy_net, beta=0.1, pad_id=tokenizer.pad_token_id)
    dpo_optimizer = optim.Adam(dpo_model.parameters(), lr=policy_lr)
    logger.debug("DPO model and optimizer initialized")  # {{ edit_3 }}
    
    # Initialize sentiment analysis pipeline for determining preferences
    preference_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if torch.cuda.is_available() else -1
    )
    logger.debug("Sentiment analysis pipeline initialized")  # {{ edit_4 }}
    
    policy_optimizer = optim.Adam(
        policy_net.parameters(), lr=policy_lr
    )
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
    logger.debug("Policy and Value optimizers initialized")  # {{ edit_6 }}
    
    # Initialize Live Dataset Generator  # {{ edit_5 }}
    live_dataset_gen = LiveDatasetGenerator(model_name=model_name)  # {{ edit_6 }}
    logger.debug("Live dataset generator initialized")  # {{ edit_6 }}
    
    for iteration in range(num_iterations):
        logger.info(
            f"Starting iteration {iteration + 1}/{num_iterations}"
        )
        for episode in range(episodes_per_iteration):
            logger.debug(
                f"Starting episode {episode + 1}/{episodes_per_iteration}"
            )
            # Generate live training examples  # {{ edit_7 }}
            training_batch = live_dataset_gen.get_batch(batch_size)
            logger.debug(f"Generated training batch with size: {len(training_batch)}")  # {{ edit_8 }}
            
            # Process the generated batch
            aggregated_task = ""
            for example in training_batch:
                user_task = example['description']  # {{ edit_66 }}
                code_files = example['code_files']  # {{ edit_67 }}
                if code_files:
                    # {{ edit_86 }} Convert code_files list to a single string
                    code_content = '\n'.join([f"File: {cf['file_path']}\n{cf['content']}" for cf in code_files])
                    aggregated_task += user_task + '\n' + code_content + '\n'
                    logger.debug(f"Combined user_task and code_files: {aggregated_task}")  # {{ edit_87 }}
                else:
                    aggregated_task += user_task + '\n'
                    logger.debug(f"Combined user_task without code_files: {aggregated_task}")  # {{ edit_87 }}
        
            # Initialize state_sequence with user_task and tokenized code files
            # Encode the aggregated user task
            task_tokens = tokenizer.encode(aggregated_task, return_tensors='pt', truncation=True, max_length=max_length).to(device)  # {{ edit_88 }}
            logger.debug("User task encoded with shape: {}".format(task_tokens.shape))  # {{ edit_7 }}
            
            # Get tokenized code files from the agent's state representation
            code_representation = agent.get_code_files()
            if code_representation:
                # {{ edit_89 }} Convert code_representation list to a single string
                code_content_agent = '\n'.join([f"File: {cf['file_path']}\n{cf['content']}" for cf in code_representation])
                code_tokens = tokenizer.encode(code_content_agent, return_tensors='pt', truncation=True, max_length=max_length).to(device)  # {{ edit_90 }}
                logger.debug("Code representation encoded with shape: {}".format(code_tokens.shape))  # {{ edit_8 }}
            else:
                logger.warning("No code_representation found from agent.")  # {{ edit_91 }}
                code_tokens = torch.zeros(1, 0).to(device)  # Empty tensor
            
            # Combine task and code tokens
            if code_tokens.size(1) > 0:
                initial_state = torch.cat((task_tokens, code_tokens), dim=1)
                logger.debug("Initial state concatenated with shape: {}".format(initial_state.shape))  # {{ edit_9 }}
            else:
                initial_state = task_tokens
                logger.debug("Initial state set to task_tokens only with shape: {}".format(initial_state.shape))  # {{ edit_92 }}
            
            state_sequence = initial_state
            logger.debug(f"Current state sequence shape: {state_sequence.shape}")  # {{ edit_54 }}
            # Ensure the sequence does not exceed the maximum length
            if state_sequence.size(1) > sequence_length:
                state_sequence = state_sequence[:, -sequence_length:]
                logger.debug("State sequence truncated to sequence_length={}".format(sequence_length))  # {{ edit_10 }}
    
            # Initialize prompt_mask
            prompt_mask = initialize_prompt_mask(state_sequence, tokenizer)
            logger.debug("Prompt mask initialized with shape: {}".format(prompt_mask.shape))  # {{ edit_11 }}
    
            thought_tree = ThoughtTree(state_sequence)
            trajectory = []
            preferred_sequences = []
            unpreferred_sequences = []
    
            # Initialize Iteration of Thoughts parameters
            num_thought_iterations = 3  # Number of thought iterations
            logger.debug("Starting Thought Iterations: {}".format(num_thought_iterations))  # {{ edit_12 }}
            for thought_iter in range(num_thought_iterations):
                logger.debug(
                    f"Thought Iteration {thought_iter + 1}/{num_thought_iterations}"
                )
                for depth in range(max_depth):
                    logger.debug(
                        f"Depth {depth + 1}/{max_depth} in Thought Iteration {thought_iter + 1}"
                    )
                    # Expand dimensions to match (sequence_length, batch_size, input_dim)
                    src = state_sequence.unsqueeze(
                        0
                    )  # Shape: (sequence_length, 1, input_dim)
                    logger.debug("Source tensor shape before policy_net: {}".format(src.shape))  # {{ edit_13 }}
                    action_logits = policy_net(src)  # Get logits instead of probabilities
                    logger.debug("Action logits obtained: {}".format(action_logits.shape))  # {{ edit_14 }}
                    m = Categorical(action_logits)
                    actions = m.sample((5,))  # Generate multiple branches
                    logger.debug("Sampled actions shape: {}".format(actions.shape))  # {{ edit_15 }}
                    rewards = []
                    log_probs = []  # Initialize list to store log_probs
    
                    generated_sequences = []  # To store sequences for preference determination
    
                    for action in actions:
                        logger.debug("Processing action: {}".format(action))  # {{ edit_16 }}
                        next_state = transition(
                            state_sequence[:, -1], action
                        )
                        logger.debug("Next state shape: {}".format(next_state.shape))  # {{ edit_17 }}
                        # Update the sequence by appending the new state
                        next_sequence = torch.cat(
                            [state_sequence, next_state.unsqueeze(0)],
                            dim=1,
                        )
                        logger.debug("Next sequence shape after concatenation: {}".format(next_sequence.shape))  # {{ edit_18 }}
                        # Ensure the sequence length does not exceed the maximum
                        if next_sequence.size(1) > sequence_length:
                            next_sequence = next_sequence[:, -sequence_length:]
                            logger.debug("Next sequence truncated to sequence_length={}".format(sequence_length))  # {{ edit_19 }}
                        rollout = monte_carlo_rollout(
                            policy_net,
                            next_sequence,
                            depth + 1,
                            max_depth,
                            sequence_length,
                        )
                        logger.debug("Rollout obtained with length: {}".format(len(rollout)))  # {{ edit_20 }}
                        total_reward = sum([r for _, r in rollout])
                        logger.debug("Total reward from rollout: {}".format(total_reward))  # {{ edit_21 }}
                        # Expand dimensions for reward model input
                        reward_input = next_sequence.unsqueeze(1)
                        logger.debug("Reward input shape: {}".format(reward_input.shape))  # {{ edit_22 }}
                        reward_estimate = reward_model(reward_input)
                        logger.debug("Reward estimate obtained: {}".format(reward_estimate.shape))  # {{ edit_23 }}
                        reward = reward_estimate.item() + total_reward
                        logger.debug("Final reward: {}".format(reward))  # {{ edit_24 }}
                        rewards.append(reward)
    
                        # Decode the sequence for preference determination
                        decoded_sequence = tokenizer.decode(next_sequence[0], skip_special_tokens=True)
                        generated_sequences.append(decoded_sequence)
                        logger.debug("Decoded sequence: {}".format(decoded_sequence))  # {{ edit_25 }}
    
                        # Update thought tree
                        thought_tree.add_child(
                            thought_tree.root, next_sequence, reward
                        )
                        logger.debug("Added child to thought_tree")  # {{ edit_26 }}
    
                        # Store the log probability of the action
                        log_prob = m.log_prob(action)
                        log_probs.append(log_prob.detach())
                        logger.debug("Stored log_prob: {}".format(log_prob.item()))  # {{ edit_27 }}
    
                    # Determine preferences based on generated sequences
                    preferred_seq, unpreferred_seq = determine_preferences(generated_sequences, preference_pipeline)  # {{ edit_50 }}
                    logger.debug("Preferred sequence: {}, Unpreferred sequence: {}".format(preferred_seq, unpreferred_seq))  # {{ edit_28 }}
                    
                    if preferred_seq is not None and unpreferred_seq is not None:
                        preferred_sequences.append(preferred_seq)
                        unpreferred_sequences.append(unpreferred_seq)
                        logger.debug("Appended preferred and unpreferred sequences")  # {{ edit_29 }}
    
                    # Select the best action based on rewards
                    if rewards:
                        best_action_index = (
                            torch.tensor(rewards).argmax().item()
                        )
                        best_action = actions[best_action_index]
                        best_reward = rewards[best_action_index]
                        best_log_prob = log_probs[best_action_index]  # Retrieve stored log_prob
                        logger.debug("Best action index: {}, Best reward: {}".format(best_action_index, best_reward))  # {{ edit_30 }}
    
                        # Convert Logits to Actions Using Tokenizer and Parsing Function
                        # Decode the best action using the tokenizer
                        action_id = best_action.item()
                        action_text = tokenizer.decode([action_id], skip_special_tokens=True)
                        logger.debug("Best action decoded: {}".format(action_text))  # {{ edit_31 }}
                        
                        # Map the action text to actionable commands
                        action_names = agent.map_response_to_actions(action_text)
                        logger.debug("Mapped action names: {}".format(action_names))  # {{ edit_32 }}
                        
                        for action_name in action_names:
                            # Create a Context object for the action
                            context = Context(
                                response="Training iteration action execution",
                                project_directory=agent.project_directory,
                                entities={}  # Populate with relevant entities if needed
                            )
                            # Execute the parsed action
                            agent.execute_action(action_name, context)
                            logger.debug("Executed action: {}".format(action_name))  # {{ edit_33 }}
                            
                            # Update state_sequence with the latest state from the agent after action execution
                            # Fetch the updated state representation from the agent
                            updated_state = agent.get_state_representation()
                            logger.debug("Updated state obtained from agent")  # {{ edit_34 }}
                            
                            # Tokenize the updated state
                            updated_state_tokens = tokenizer.encode(updated_state, return_tensors='pt').to(device)
                            logger.debug("Updated state tokenized with shape: {}".format(updated_state_tokens.shape))  # {{ edit_35 }}
                            
                            # Combine the existing state_sequence with the updated state
                            state_sequence = torch.cat([state_sequence, updated_state_tokens], dim=1)
                            logger.debug("State sequence updated shape: {}".format(state_sequence.shape))  # {{ edit_36 }}
                            
                            # Ensure the state_sequence does not exceed the maximum length
                            if state_sequence.size(1) > sequence_length:
                                state_sequence = state_sequence[:, -sequence_length:]
                                logger.debug("State sequence truncated to sequence_length={}".format(sequence_length))  # {{ edit_37 }}
    
                        # Store the experience with old_log_prob
                        trajectory.append(
                            (state_sequence.clone(), best_action, best_reward, best_log_prob)  # Updated reward
                        )
                        logger.debug("Appended to trajectory")  # {{ edit_38 }}
    
                # End of thought iterations
    
            # Compute DPO-specific loss using the DPO model's forward pass  # {{ edit_51 }}
            preferred_tensor = tokenizer.encode(preferred_sequences, return_tensors='pt').to(device)
            unpreferred_tensor = tokenizer.encode(unpreferred_sequences, return_tensors='pt').to(device)
            dpo_loss = dpo_model(
                preferred_seq=preferred_tensor,
                unpreferred_seq=unpreferred_tensor,
                prompt_mask=prompt_mask
            )
            logger.debug("DPO loss computed using DPO model's forward pass")  # {{ edit_52 }}
    
            # Update policy network using DPO optimizer
            dpo_optimizer.zero_grad()
            dpo_loss.backward()
            dpo_optimizer.step()
            logger.debug("DPO optimizer stepped")  # {{ edit_40 }}
    
            # Update value network as usual
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
                logger.debug("Computed advantage: {}".format(advantage))  # {{ edit_41 }}
    
            advantages_tensor = torch.tensor(
                advantages, dtype=torch.float32
            ).to(device)
            if advantages_tensor.std() != 0:
                advantages_tensor = (
                    advantages_tensor - advantages_tensor.mean()
                ) / (advantages_tensor.std() + 1e-8)
            else:
                advantages_tensor = advantages_tensor - advantages_tensor.mean()
            logger.debug("Advantages tensor prepared")  # {{ edit_42 }}
        
            # Update policy network using PPO
            policy_losses = []  # Collect policy losses for averaging
            for i, (state_seq_t, action_t, _, old_log_prob) in enumerate(
                trajectory
            ):
                # Expand dimensions to match (batch_size, sequence_length, input_dim)
                src = state_seq_t.unsqueeze(0)  # Changed to batch_first
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
                logger.debug("Policy optimizer stepped with total_policy_loss: {}".format(total_policy_loss.item()))  # {{ edit_43 }}
    
            # Update value network
            returns_tensor = (
                torch.tensor(returns, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            # Prepare inputs for the value network
            if trajectory:
                value_inputs = torch.stack(
                    [s for s, _, _, _ in trajectory]
                ).transpose(0, 1)
                value_inputs = value_inputs.to(device)
                values = value_net(value_inputs)
                value_loss = nn.MSELoss()(values, returns_tensor)
                logger.debug("Value loss computed: {}".format(value_loss.item()))  # {{ edit_44 }}
        
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
                logger.debug("Value optimizer stepped with loss: {}".format(value_loss.item()))  # {{ edit_85 }}
            else:
                logger.warning("No trajectory to update value network.")  # {{ edit_93 }}
    
            # Log the value loss if available
            if trajectory:
                logger.debug(f"Value loss: {value_loss.item()}")  # Existing debug
    
        
        if (iteration + 1) % save_interval == 0:
            save_checkpoint(policy_net, value_net, reward_model, iteration + 1)  # Retained value_net
            logger.info(f"Checkpoint saved at iteration {iteration + 1}")  # {{ edit_46 }}
        logger.info(
            f"Completed iteration {iteration + 1}/{num_iterations}"
        )
        logger.debug(f"Iteration {iteration + 1} completed") 


if __name__ == "__main__":
    # Initialize the SoftwareEngineeringAgent
    agent = SoftwareEngineeringAgent(project_directory='virtual_env')  
    
    input_dim: int = 768
    action_dim: int = 10
    nhead: int = 8  
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    eos_token_id: int = 102
    # Initialize the policy, value, and reward networks
    policy_net = TransformerPolicyNetwork(
            input_dim,
            action_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device) 
    value_net = TransformerValueNetwork(
            input_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)
    reward_model = TransformerRewardModel(
            input_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(device)
    
    # Start training
    train(
        agent=agent,
        policy_net=policy_net,
        value_net=value_net,
        reward_model=reward_model,
        num_iterations=1000,
        episodes_per_iteration=10,
        max_depth=5,
        sequence_length=10,
        gamma=0.99,
        clip_epsilon=0.2,
        policy_lr=1e-4,
        value_lr=1e-3,
        save_interval=20,
        batch_size=4,
        max_length=8192,
        reward_shaping_factor=1.0,
        reward_shaping_success=0.5,
        reward_shaping_failure=-0.5,
        model_name='gpt2',
    )



