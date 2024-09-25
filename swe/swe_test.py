import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from open_strawberry_torch.model import TransformerPolicyNetwork, TransformerValueNetwork, TransformerRewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_forward_pass():
    # Define input dimensions based on your model's expectations
    batch_size = 2
    sequence_length = 10
    input_dim = 768  # As per your SoftwareEngineeringAgent

    # Create dummy input data
    dummy_input = torch.zeros(sequence_length, batch_size, input_dim).to(device)

    # Initialize models
    policy_net = TransformerPolicyNetwork(input_dim=input_dim, action_dim=10).to(device)
    value_net = TransformerValueNetwork(input_dim=input_dim).to(device)
    reward_model = TransformerRewardModel(input_dim=input_dim).to(device)

    # Forward pass through Policy Network
    policy_output = policy_net(dummy_input)
    assert policy_output.shape == (sequence_length, batch_size, 10), f"Policy output shape mismatch: {policy_output.shape}"
    
    # Change the assertion to accommodate the actual output shape
    expected_shape = (sequence_length, batch_size, 10)
    actual_shape = policy_output.shape
    if actual_shape != expected_shape:
        raise AssertionError(f"Policy output shape mismatch: expected {expected_shape}, got {actual_shape}")

    print("Forward pass tests passed.")

def test_parameter_initialization():
    policy_net = TransformerPolicyNetwork(input_dim=768, action_dim=10).to(device)
    value_net = TransformerValueNetwork(input_dim=768).to(device)
    reward_model = TransformerRewardModel(input_dim=768).to(device)

    for name, param in policy_net.named_parameters():
        assert not torch.isnan(param).any(), f"NaN found in {name}"
        assert not torch.isinf(param).any(), f"Inf found in {name}"

    for name, param in value_net.named_parameters():
        assert not torch.isnan(param).any(), f"NaN found in {name}"
        assert not torch.isinf(param).any(), f"Inf found in {name}"

    for name, param in reward_model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN found in {name}"
        assert not torch.isinf(param).any(), f"Inf found in {name}"

    print("Parameter initialization tests passed.")

def test_output_sanity():
    batch_size = 2
    sequence_length = 10
    input_dim = 768

    dummy_input = torch.randn(sequence_length, batch_size, input_dim).to(device)

    policy_net = TransformerPolicyNetwork(input_dim=input_dim, action_dim=10).to(device)
    policy_output = policy_net(dummy_input)  # Assuming output shape: (sequence_length, batch_size, action_dim)

    # Check if probabilities sum to 1
    probs_sum = policy_output.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5), "Policy probabilities do not sum to 1."

    print("Output sanity checks passed.")

def test_gradient_flow():
    batch_size = 2
    sequence_length = 10
    input_dim = 768
    action_dim = 10

    dummy_input = torch.randn(sequence_length, batch_size, input_dim).to(device)
    dummy_labels = torch.randint(0, action_dim, (sequence_length, batch_size)).to(device)

    policy_net = TransformerPolicyNetwork(input_dim=input_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # Forward pass
    policy_output = policy_net(dummy_input)  # Output shape: (sequence_length, batch_size, action_dim)
    loss_fn = nn.CrossEntropyLoss()

    # Reshape for loss computation
    policy_output_flat = policy_output.view(-1, action_dim)
    dummy_labels_flat = dummy_labels.view(-1)

    loss = loss_fn(policy_output_flat, dummy_labels_flat)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check if gradients are not None
    for name, param in policy_net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("Gradient flow tests passed.")

from torch.utils.data import DataLoader, TensorDataset

def test_training_on_synthetic_data():
    # Create synthetic data
    input_dim = 768
    action_dim = 10
    sequence_length = 10
    batch_size = 2

    # Simple task: predict the next token as a specific class
    X = torch.randn(sequence_length, batch_size, input_dim).to(device)
    y = torch.randint(0, action_dim, (sequence_length, batch_size)).to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=1)

    # Initialize model and optimizer
    policy_net = TransformerPolicyNetwork(input_dim=input_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    policy_net.train()
    for epoch in range(5):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = policy_net(batch_x)  # Shape: (sequence_length, batch_size, action_dim)
            output_flat = output.view(-1, action_dim)
            batch_y_flat = batch_y.view(-1)
            loss = loss_fn(output_flat, batch_y_flat)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # After training, check if loss has decreased
    assert loss.item() < 2.0, "Loss did not decrease as expected on synthetic data."

    print("Integration test with synthetic data passed.")

def test_on_real_data():
    # Assuming SwoDataset and related components are correctly implemented
    data_path = 'path_to_small_subset'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_loader, val_loader = SwoDataset.load_dataset(data_path, tokenizer, batch_size=2, max_length=512)

    # Get a single batch
    for batch in train_loader:
        # Assuming batch contains input_ids and labels
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # Forward pass through Policy Network
        policy_net = TransformerPolicyNetwork(input_dim=768, action_dim=10).to(device)
        policy_output = policy_net(input_ids)

        # Check output shape
        assert policy_output.shape == (input_ids.size(0), 10), f"Policy output shape mismatch: {policy_output.shape}"
        print("Real data forward pass test passed.")
        break  # Test on a single batch

test_on_real_data()


test_forward_pass()
test_parameter_initialization()
test_output_sanity()
test_gradient_flow()
test_training_on_synthetic_data()
