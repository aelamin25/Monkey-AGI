import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from .network import ActorCriticNet

@dataclass
class Trajectory:
    observations: list
    actions: list
    rewards: list

class MonkeyAgent:
    def __init__(self, lr=1e-2, gamma=0.99, device="cpu"):
        self.gamma = gamma
        self.device = device
        self.net = ActorCriticNet().to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def choose_action(self, obs):
        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, _ = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action, probs[0, 0].item()

    def update(self, traj: Trajectory):
        obs = torch.tensor(np.vstack(traj.observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(traj.actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(traj.rewards, dtype=torch.float32, device=self.device)

        T = len(rewards)
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        logits, values = self.net(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_actions = log_probs[range(T), actions]

        advantages = returns - values.detach()

        policy_loss = -(log_probs_actions * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()
