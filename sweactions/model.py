import torch.nn as nn

class TransformerPolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, batch_first=False):
        super(TransformerPolicyNetwork, self).__init__()
        self.transformer = nn.Transformer(batch_first=batch_first)
        self.fc = nn.Linear(input_dim, action_dim)
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return x