import json
import matplotlib.pyplot as plt
import argparse


def run_RBPPO_plot_reward(arg_role, arg_shaping):
    if arg_role == "ice" or arg_role == "Ice":
        role_string = "ice"
    if arg_role == "ore" or arg_role == "Ore":
        role_string = "ore"
    
    if arg_shaping == "light" or arg_shaping == "Light":
        shaping_string = "light"
    if arg_shaping == "heavy" or arg_shaping == "Heavy":
        shaping_string = "heavy"
    
    with open("training_data/avg_training_rewards_" + role_string + "_" + shaping_string + ".json", "r") as f:
        rewards = json.load(f)

    plt.plot(rewards)
    plt.xlabel("Training iteration (10000 steps each)")
    plt.ylabel("Average episodic reward")
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig("training_graphs/ppo_training_plot_" + role_string + "_" + shaping_string + "_reward.png")
    plt.close()
   
    
def run_RBPPO_plot_KL(arg_role, arg_shaping):
    if arg_role == "ice" or arg_role == "Ice":
        role_string = "ice"
    if arg_role == "ore" or arg_role == "Ore":
        role_string = "ore"
    
    if arg_shaping == "light" or arg_shaping == "Light":
        shaping_string = "light"
    if arg_shaping == "heavy" or arg_shaping == "Heavy":
        shaping_string = "heavy"
    
    with open("training_data/avg_training_kl_" + role_string + "_" + shaping_string + ".json", "r") as f:
        kl = json.load(f)

    plt.plot(kl)
    plt.xlabel("Training iteration (10000 steps each)")
    plt.ylabel("Average episodic KL")
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig("training_graphs/ppo_training_plot_" + role_string + "_" + shaping_string + "_kl.png")
    plt.close()
    
    
def run_RBPPO_plot_entropy(arg_role, arg_shaping):
    if arg_role == "ice" or arg_role == "Ice":
        role_string = "ice"
    if arg_role == "ore" or arg_role == "Ore":
        role_string = "ore"
    
    if arg_shaping == "light" or arg_shaping == "Light":
        shaping_string = "light"
    if arg_shaping == "heavy" or arg_shaping == "Heavy":
        shaping_string = "heavy"
    
    with open("training_data/avg_training_entropy_" + role_string + "_" + shaping_string + ".json", "r") as f:
        entropy = json.load(f)

    plt.plot(entropy)
    plt.xlabel("Training iteration (10000 steps each)")
    plt.ylabel("Average episodic entropy")
    plt.title("Training Progress")
    plt.grid(True)
    plt.savefig("training_graphs/ppo_training_plot_" + role_string + "_" + shaping_string + "_entropy.png")
    plt.close()
    
    
# Execute the training code with parsed arguments (caps insensitive) - usage: python RBPPO_ppo.py --role "ice" --shaping "heavy"
parser = argparse.ArgumentParser(description="Run training data plotting script for designated type.")
parser.add_argument("--role", choices=["Ice", "ice", "Ore", "ore"], required=True, help="Choose which role to plot (Ice Miner or Ore Miner).")
parser.add_argument("--shaping", choices=["Light", "light", "Heavy", "heavy"], required=True, help="Reward shaping intensity to plot.")

args = parser.parse_args()
run_RBPPO_plot_reward(args.role, args.shaping)
run_RBPPO_plot_KL(args.role, args.shaping)
run_RBPPO_plot_entropy(args.role, args.shaping)