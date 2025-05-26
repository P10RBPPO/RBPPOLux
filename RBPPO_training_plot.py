import json
import matplotlib.pyplot as plt
import argparse
import os


def run_RBPPO_plot_combined_rewards(arg_role, shaping_level, shaping_name):
    """
    Generates a combined plot for the given shaping level and all lower levels for rewards.
    """
    shaping_map = {0: "simple", 1: "moderate", 2: "complex"}
    role_string = "ice" if arg_role.lower() == "ice" else "ore"
    levels = list(range(shaping_level + 1))  # Include the current level and all lower levels
    plt.figure(figsize=(10, 6))  # Increase figure size to 1000x600 pixels

    for level in levels:
        file_path = f"training_data/avg_training_rewards_{level}_{role_string}_{shaping_name}.json"
        if not os.path.exists(file_path):
            print(f"Missing training data for {role_string.capitalize()} {shaping_name.capitalize()} (Level {level}) in training_data folder.")
            continue

        with open(file_path, "r") as f:
            rewards = json.load(f)

        shaping_name_corrected = shaping_map.get(level, shaping_name)
        plt.plot(rewards, label=f"Level {level} ({shaping_name_corrected.capitalize()})")  # Add a label for the legend

    plt.xlabel("Training iteration (10000 steps each)", fontsize=14)
    plt.ylabel("Average episodic reward", fontsize=14)
    plt.title(f"Training Progress - Role: {role_string.capitalize()}, Shaping: {shaping_name.capitalize()} (Level {shaping_level})", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)  # Add a legend to differentiate levels
    os.makedirs("training_graphs", exist_ok=True)
    plt.savefig(f"training_graphs/ppo_training_plot_{role_string}_{shaping_name}_level_{shaping_level}_combined_rewards.png", dpi=300)
    plt.close()


def run_RBPPO_plot_reward(arg_role, shaping_level, shaping_name):
    """
    Generates a separate plot for the average reward of a specific shaping level and category.
    """
    role_string = "ice" if arg_role.lower() == "ice" else "ore"
    file_path = f"training_data/avg_training_rewards_{shaping_level}_{role_string}_{shaping_name}.json"

    if not os.path.exists(file_path):
        print(f"Missing training data for {role_string.capitalize()} {shaping_name.capitalize()} (Level {shaping_level}) in training_data folder.")
        return

    with open(file_path, "r") as f:
        rewards = json.load(f)

    plt.figure(figsize=(10, 6))  # Increase figure size to 1000x600 pixels
    plt.plot(rewards)
    plt.xlabel("Training iteration (10000 steps each)", fontsize=14)
    plt.ylabel("Average episodic reward", fontsize=14)
    plt.title(f"Training Progress\nRole: {role_string.capitalize()}, Shaping: {shaping_name.capitalize()} (Level {shaping_level})", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    os.makedirs("training_graphs", exist_ok=True)
    plt.savefig(f"training_graphs/ppo_training_plot_{role_string}_{shaping_name}_level_{shaping_level}_reward.png", dpi=300)
    plt.close()


def run_all_combinations():
    roles = ["ice", "ore"]
    shapings = [("simple", 0), ("moderate", 1), ("complex", 2)]  # Shaping names and their corresponding levels
    for role in roles:
        for shaping_name, shaping_level in shapings:
            print(f"Generating combined plot for role: {role}, shaping: {shaping_name} (Level {shaping_level})")
            run_RBPPO_plot_combined_rewards(role, shaping_level, shaping_name)
            print(f"Generating separate plot for role: {role}, shaping: {shaping_name} (Level {shaping_level})")
            run_RBPPO_plot_reward(role, shaping_level, shaping_name)


# Create the training_graphs directory if it doesn't exist
os.makedirs("training_graphs", exist_ok=True)

# Execute the training code with parsed arguments
parser = argparse.ArgumentParser(description="Run training data plotting script for designated type.")
parser.add_argument("--role", choices=["Ice", "ice", "Ore", "ore"], help="Choose which role to plot (Ice Miner or Ore Miner).")
parser.add_argument("--shaping", choices=["Simple", "simple", "Moderate", "moderate", "Complex", "complex"], help="Reward shaping intensity to plot.")
parser.add_argument("--all", action="store_true", help="Generate graphs for all role and shaping combinations.")

args = parser.parse_args()

if args.all:
    run_all_combinations()
else:
    if args.role and args.shaping:
        shaping_map = {"simple": 0, "moderate": 1, "complex": 2}
        shaping_name = args.shaping.lower()
        shaping_level = shaping_map[shaping_name]
        run_RBPPO_plot_combined_rewards(args.role, shaping_level, shaping_name)
        run_RBPPO_plot_reward(args.role, shaping_level, shaping_name)
    else:
        print("Error: You must specify both --role and --shaping, or use --all to generate all plots.")