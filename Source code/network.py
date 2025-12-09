import torch
import torch.nn as nn

class ActorCriticNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, n_actions=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = torch.tanh(self.fc(x))
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value
