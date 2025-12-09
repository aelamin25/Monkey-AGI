import matplotlib.pyplot as plt

def render_scene(ax, env, last_action, last_reward, prob_left, day):
    ax.clear()
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1.5, 1)
    ax.axis("off")

    # Taps
    for i, x in enumerate([0.0, 1.0]):
        face = "red" if i == 0 else "blue"
        label = ("LEFT" if i == 0 else "RIGHT")
        if i == env.sugar_tap:
            label += " (sugar)"
        rect = plt.Rectangle((x-0.3, 0.0), 0.6, 0.4, edgecolor="black", facecolor=face, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, 0.45, label, ha="center")

    # Monkey
    if last_action is not None:
        ax.text([0.0, 1.0][last_action], -0.6, "🐒", fontsize=36, ha="center")
        if last_reward is not None:
            ax.text([0.0,1.0][last_action], -0.15, f"reward={last_reward}", ha="center")

    ax.set_title(f"Day {day} | π(left)={prob_left:.2f}")

def render_reward(ax, days, rewards):
    ax.clear()
    ax.plot(days, rewards, color="blue")
    ax.set_title("Average reward")
    ax.set_xlabel("Day")

def render_prob(ax, days, probs):
    ax.clear()
    ax.plot(days, probs, color="orange")
    ax.set_title("Policy π(left)")
    ax.set_xlabel("Day")
    ax.set_ylim(0, 1)
