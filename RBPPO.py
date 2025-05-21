import argparse
import json
import math
import os

from RBPPO_lux_env import LuxCustomEnv
from RBPPO_ppo import PPO


def run_RBPPO(role_type, shaping_type):
    if role_type == "ice" or role_type == "Ice":
        role_string = "ice"
        role_index = 0
    if role_type == "ore" or role_type == "Ore":
        role_string = "ore"
        role_index = 1
    
    if shaping_type == "light" or shaping_type == "Light":
        heavy_shaping = False
        shaping_string = "light"
    if shaping_type == "heavy" or shaping_type == "Heavy":
        heavy_shaping = True
        shaping_string = "heavy"
    
    avg_reward_log_path = "training_data/avg_training_rewards_" + role_string + "_" + shaping_string + ".json"
    avg_reward_history = []
    avg_kl_log_path = "training_data/avg_training_kl_" + role_string + "_" + shaping_string + ".json"
    avg_kl_history = []
    avg_entropy_log_path = "training_data/avg_training_entropy_" + role_string + "_" + shaping_string + ".json"
    avg_entropy_history = []

    # Load previous logs if available
    if os.path.exists(avg_reward_log_path):
        with open(avg_reward_log_path, "r") as f:
            avg_reward_history = json.load(f)
    if os.path.exists(avg_kl_log_path):
        with open(avg_kl_log_path, "r") as f:
            avg_kl_history = json.load(f)
    if os.path.exists(avg_entropy_log_path):
        with open(avg_entropy_log_path, "r") as f:
            avg_entropy_history = json.load(f)

    env = LuxCustomEnv()
    model = PPO(env, role_index, heavy_shaping) # params: env, role_index [0, 1], heavy_shaping_param [True, False]
    epoch = 0
    
    try:
        model.load(False, "models/rbppo_checkpoint_" + role_string + "_" + shaping_string + ".pth")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh.")

    epoch = model.epoch
    while True:
        avg_reward, avg_kl, avg_entropy = model.learn(10000, False)
        avg_reward_history.append(avg_reward)
        avg_kl_history.append(avg_kl)
        avg_entropy_history.append(avg_entropy)
        
        # Save model
        epoch += 10000
        model.save(epoch, False, "models/rbppo_checkpoint_" + role_string + "_" + shaping_string + ".pth")
        
        # Save model at certain milestones in their own files
        if (epoch > 0) and (math.log10(epoch) % 1 == 0):
            model.save(epoch, False, "models/rbppo_checkpoint_" + role_string + "_" + shaping_string + "_" + str(epoch) + ".pth")
        
        # Save log information (reward, kl, entropy)
        with open(avg_reward_log_path, "w") as f:
            json.dump(avg_reward_history, f)
        with open(avg_kl_log_path, "w") as f:
            json.dump(avg_kl_history, f)
        with open(avg_entropy_log_path, "w") as f:
            json.dump(avg_entropy_history, f)

# Execute the training code with parsed arguments (caps insensitive) - usage: python RBPPO_ppo.py --role "ice" --shaping "heavy"
parser = argparse.ArgumentParser(description="Run RBPPO training with specific role and shaping type.")
parser.add_argument("--role", choices=["Ice", "ice", "Ore", "ore"], required=True, help="Choose which role to train (Ice Miner or Ore Miner).")
parser.add_argument("--shaping", choices=["Light", "light", "Heavy", "heavy"], required=True, help="Reward shaping intensity.")

args = parser.parse_args()
run_RBPPO(args.role, args.shaping)
