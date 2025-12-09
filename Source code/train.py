import numpy as np
import matplotlib.pyplot as plt
from .env import TapEnv
from .agent import MonkeyAgent, Trajectory
from .visualize import render_scene, render_reward, render_prob

def train(num_days=300, switch_after=None):
    env = TapEnv(sugar_tap=0)
    agent = MonkeyAgent()

    plt.ion()
    fig, (ax_scene, ax_reward, ax_prob) = plt.subplots(3, 1, figsize=(7, 10))
    fig.tight_layout(pad=3.0)

    days, avg_rewards, probs = [], [], []
    window = []

    last_action = None
    last_reward = None
    last_prob_left = 0.5

    for day in range(1, num_days+1):

        if switch_after and day == switch_after:
            env.switch_sugar()

        obs = env.reset()
        traj = Trajectory([], [], [])
        ep_reward = 0.0

        action, prob_left = agent.choose_action(obs)
        obs_next, reward, done, _ = env.step(action)

        traj.observations.append(obs)
        traj.actions.append(action)
        traj.rewards.append(reward)

        ep_reward += reward
        last_action, last_reward, last_prob_left = action, reward, prob_left

        agent.update(traj)

        window.append(ep_reward)
        if len(window) > 20:
            window.pop(0)
        avg_reward = float(np.mean(window))

        days.append(day)
        avg_rewards.append(avg_reward)
        probs.append(last_prob_left)

        render_scene(ax_scene, env, last_action, last_reward, last_prob_left, day)
        render_reward(ax_reward, days, avg_rewards)
        render_prob(ax_prob, days, probs)
        plt.pause(0.01)

    plt.ioff()
