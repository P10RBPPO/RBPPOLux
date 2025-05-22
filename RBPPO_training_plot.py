import json
import matplotlib.pyplot as plt
import argparse
import os


def run_RBPPO_plot_reward(arg_role, arg_shaping):
    role_string = "ice" if arg_role.lower() == "ice" else "ore"
    shaping_string = "light" if arg_shaping.lower() == "light" else "heavy"
    file_path = f"training_data/avg_training_rewards_{role_string}_{shaping_string}.json"

    if not os.path.exists(file_path):
        print(f"Missing training data for {role_string.capitalize()} {shaping_string.capitalize()} in RBPPOLux/training_data")
        return

    with open(file_path, "r") as f:
        rewards = json.load(f)

    plt.figure(figsize=(10, 6))  # Increase figure size to 1000x600 pixels
    plt.plot(rewards)
    plt.xlabel("Training iteration (10000 steps each)", fontsize=14)
    plt.ylabel("Average episodic reward", fontsize=14)
    plt.title(f"Training Progress\nRole: {role_string.capitalize()}, Shaping: {shaping_string.capitalize()}", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig(f"training_graphs/ppo_training_plot_{role_string}_{shaping_string}_reward.png", dpi=300)
    plt.close()


def run_RBPPO_plot_KL(arg_role, arg_shaping):
    role_string = "ice" if arg_role.lower() == "ice" else "ore"
    shaping_string = "light" if arg_shaping.lower() == "light" else "heavy"
    file_path = f"training_data/avg_training_kl_{role_string}_{shaping_string}.json"

    if not os.path.exists(file_path):
        print(f"Missing training data for {role_string.capitalize()} {shaping_string.capitalize()} in RBPPOLux/training_data")
        return

    with open(file_path, "r") as f:
        kl = json.load(f)

    plt.figure(figsize=(10, 6))  # Increase figure size to 1000x600 pixels
    plt.plot(kl)
    plt.xlabel("Training iteration (10000 steps each)", fontsize=14)
    plt.ylabel("Average episodic KL", fontsize=14)
    plt.title(f"Training Progress\nRole: {role_string.capitalize()}, Shaping: {shaping_string.capitalize()}", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig(f"training_graphs/ppo_training_plot_{role_string}_{shaping_string}_kl.png", dpi=300)
    plt.close()


def run_RBPPO_plot_entropy(arg_role, arg_shaping):
    role_string = "ice" if arg_role.lower() == "ice" else "ore"
    shaping_string = "light" if arg_shaping.lower() == "light" else "heavy"
    file_path = f"training_data/avg_training_entropy_{role_string}_{shaping_string}.json"

    if not os.path.exists(file_path):
        print(f"Missing training data for {role_string.capitalize()} {shaping_string.capitalize()} in RBPPOLux/training_data")
        return

    with open(file_path, "r") as f:
        entropy = json.load(f)

    plt.figure(figsize=(10, 6))  # Increase figure size to 1000x600 pixels
    plt.plot(entropy)
    plt.xlabel("Training iteration (10000 steps each)", fontsize=14)
    plt.ylabel("Average episodic entropy", fontsize=14)
    plt.title(f"Training Progress\nRole: {role_string.capitalize()}, Shaping: {shaping_string.capitalize()}", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.savefig(f"training_graphs/ppo_training_plot_{role_string}_{shaping_string}_entropy.png", dpi=300)
    plt.close()


def run_all_combinations():
    roles = ["ice", "ore"]
    shapings = ["light", "heavy"]
    for role in roles:
        for shaping in shapings:
            print(f"Generating plots for role: {role}, shaping: {shaping}")
            run_RBPPO_plot_reward(role, shaping)
            run_RBPPO_plot_KL(role, shaping)
            run_RBPPO_plot_entropy(role, shaping)


# Create the training_graphs directory if it doesn't exist
os.makedirs("training_graphs", exist_ok=True)

# Execute the training code with parsed arguments
parser = argparse.ArgumentParser(description="Run training data plotting script for designated type.")
parser.add_argument("--role", choices=["Ice", "ice", "Ore", "ore"], help="Choose which role to plot (Ice Miner or Ore Miner).")
parser.add_argument("--shaping", choices=["Light", "light", "Heavy", "heavy"], help="Reward shaping intensity to plot.")
parser.add_argument("--all", action="store_true", help="Generate graphs for all role and shaping combinations.")

args = parser.parse_args()

if args.all:
    run_all_combinations()
else:
    if args.role and args.shaping:
        run_RBPPO_plot_reward(args.role, args.shaping)
        run_RBPPO_plot_KL(args.role, args.shaping)
        run_RBPPO_plot_entropy(args.role, args.shaping)
    else:
        print("Error: You must specify both --role and --shaping, or use --all to generate all plots.")