import argparse
import json
import math
import os

from RBPPO_lux_env import LuxCustomEnv
from RBPPO_ppo import PPO


def run_RBPPO(role_type, shaping_type):
    if role_type.lower() == "ice":
        role_string = "ice"
        role_index = 0
    if role_type.lower() == "ore":
        role_string = "ore"
        role_index = 1
    
    if shaping_type.lower() == "simple":
        shaping_level = 0
        shaping_string = "simple"
    if shaping_type.lower() == "moderate":
        shaping_level = 1
        shaping_string = "moderate"
    if shaping_type.lower() == "complex":
        shaping_level = 2
        shaping_string = "complex"
    
    avg_reward_log_path = "training_data/avg_training_rewards_" + role_string + "_" + shaping_string + ".json"
    avg_reward_history = []
    avg_kl_log_path = "training_data/avg_training_kl_" + role_string + "_" + shaping_string + ".json"
    avg_kl_history = []
    avg_entropy_log_path = "training_data/avg_training_entropy_" + role_string + "_" + shaping_string + ".json"
    avg_entropy_history = []
    
    avg_reward_log_path_0 = "training_data/avg_training_rewards_0" + role_string + "_" + shaping_string + ".json"
    avg_reward_history_0 = []
    avg_reward_log_path_1 = "training_data/avg_training_rewards_1" + role_string + "_" + shaping_string + ".json"
    avg_reward_history_1 = []
    avg_reward_log_path_2 = "training_data/avg_training_rewards_2" + role_string + "_" + shaping_string + ".json"
    avg_reward_history_2 = []

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
            
    if os.path.exists(avg_reward_log_path_0):
        with open(avg_reward_log_path_0, "r") as f:
            avg_reward_history_0 = json.load(f)
    if os.path.exists(avg_reward_log_path_1):
        with open(avg_reward_log_path_1, "r") as f:
            avg_reward_history_1 = json.load(f)
    if os.path.exists(avg_reward_log_path_2):
        with open(avg_reward_log_path_2, "r") as f:
            avg_reward_history_2 = json.load(f)

    env = LuxCustomEnv()
    model = PPO(env, role_index, shaping_level) # params: env, role_index [0, 1], heavy_shaping_param [True, False]
    epoch = 0
    
    try:
        model.load(False, "models/rbppo_checkpoint_" + role_string + "_" + shaping_string + ".pth")
    except FileNotFoundError:
        print("No checkpoint found, starting fresh.")

    epoch = model.epoch
    while True:
        avg_reward, avg_kl, avg_entropy, avg_reward_0, avg_reward_1, avg_reward_2 = model.learn(10000, False)
        avg_reward_history.append(avg_reward)
        avg_kl_history.append(avg_kl)
        avg_entropy_history.append(avg_entropy)
        
        avg_reward_history_0.append(avg_reward_0)
        avg_reward_history_1.append(avg_reward_1)
        avg_reward_history_2.append(avg_reward_2)
        
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
            
        with open(avg_reward_log_path_0, "w") as f:
            json.dump(avg_reward_history_0, f)
        with open(avg_reward_log_path_1, "w") as f:
            json.dump(avg_reward_history_1, f)
        with open(avg_reward_log_path_2, "w") as f:
            json.dump(avg_reward_history_2, f)


# Execute the training code with parsed arguments (caps insensitive) - usage: python RBPPO_ppo.py --role "ice" --shaping "heavy"
parser = argparse.ArgumentParser(description="Run RBPPO training with specific role and shaping type.")
parser.add_argument("--role", choices=["ice", "ore"], required=True, help="Choose which role to train (Ice Miner or Ore Miner).")
parser.add_argument("--shaping", choices=["simple", "moderate", "complex"], required=True, help="Reward shaping intensity.")

args = parser.parse_args()
run_RBPPO(args.role, args.shaping)
