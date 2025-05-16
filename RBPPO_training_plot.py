import json
import matplotlib.pyplot as plt

with open("training_rewards.json", "r") as f:
    rewards = json.load(f)

plt.plot(rewards)
plt.xlabel("Training iteration (1000 steps each)")
plt.ylabel("Average episodic reward")
plt.title("Training Progress")
plt.grid(True)
plt.savefig("ppo_training_plot.png")