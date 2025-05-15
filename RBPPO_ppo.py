import torch
import numpy as np
import copy
import gymnasium as gym
import os
import json

from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn.functional as F

from RBPPO_lux_obs_parser import obs_parser
from RBPPO_network import FeedForwardNN
from RBPPO_lux_env import LuxCustomEnv

from lux.kit import obs_to_game_state, GameState
from RBPPO_lux_action_parser import parse_all_actions, factory_action_parser

from Controllers.FactoryController import FactoryController
from Controllers.RobotController import RobotController

class PPO:
    def __init__(self, env):
        # If GPU, then use it, else use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init role array
        self.roles = ["Ice Miner", "Ore Miner", "Rubble Cleaner"]
        
        # Init hyperparams
        self._init_hyperparameters()
        
        # Get environment information
        self.env = env
        
        # Observation space and Action space dimension definitions 
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n
        
        # Init actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        # Move networks to the GPU
        #self.actor.to(self.device)
        #self.critic.to(self.device)
        
        # Init Optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
    def _init_hyperparameters(self):
        #  Default values for now
        self.timesteps_per_batch = 2000         # Timesteps per batch
        self.max_timesteps_per_episode = 1000   # Timesteps per episode    
        
        self.n_updates_per_iteration = 5        # Epoch count
        self.clip = 0.2                         # Recommended clip threshold for PPO in PPO paper
        self.gamma = 0.95                       # Discount value
        self.lr = 0.005                         # Learning rate
        
        # Optimization hyperparams
        self.max_grad_norm = 0.5                # Gradient clipping value
        self.num_minibatches = 5                # Minibatch size
        self.ent_coef = 0                       # Entropy coefficient for Entropy Regularization
        self.target_kl = 0.02                   # KL Divergence threshold
        self.lam = 0.98                         # Lambda parameter for GAE
        
        self.player = "player_0"                # Player identifier
        self.role = self.roles[0]               # Desired role for training
    
    def learn(self, total_timesteps):
        t_so_far = 0 # Timestep counter
        
        batch_rews_all = [] # reward storage for full training batch
        
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
            
            # Save batch rewards
            batch_rews_all.extend(batch_rews)
            
            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 
            V = self.critic(batch_obs).squeeze(-1)
            batch_rtgs = A_k + V.detach()   
            
            # Calculate collected timesteps for this batch
            t_so_far += np.sum(batch_lens)
            
            # Normalize advantages - decreases variance of advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            # Minibatch
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            loss = []
            
            for _ in range(self.n_updates_per_iteration):
                # Learning rate annealing - Dynamic learning rate that changes over time
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                # Ensure it cannot drop below 0
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr
                
                # Mini-batch Update
                np.random.shuffle(inds) # Shuffle index
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    # Extract data at sampled indicies
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_probs = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]
                    
                    # Calculate V_phi and pi_theta(a_t | s_t)
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)

                    # Calculate ratios
                    logratios = curr_log_probs - mini_log_probs
                    ratios = torch.exp(logratios)
                    approx_kl = ((ratios - 1) - logratios).mean()
                    
                    # Calculate surrogate losses
                    surr1 = ratios * mini_advantage
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
                    
                    # Calculate actor loss
                    # Negative value because we want to maximize actor loss for SGD
                    # Adam optimizer minimizes overall loss
                    # Minimizing negative loss maximizes performance function
                    # Mean generates single loss as float
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    
                    # MSE loss for critic network
                    critic_loss = nn.MSELoss()(V.view(-1), mini_rtgs.view(-1))
                
                    # Entrophy Regularization
                    entropy_loss = entropy.mean()
                    # Discount entropy loss by given coefficient
                    actor_loss = actor_loss - self.ent_coef * entropy_loss
                    
                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True) # retain_graph=True
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # Gradient clipping to L2 norm
                    self.actor_optim.step()
                    
                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm) # Gradient clipping to L2 norm
                    self.critic_optim.step()
                    
                    loss.append(actor_loss.detach())
                
                # Approximating KL Divergence
                if approx_kl > self.target_kl:
                    break # if KL is above threshold

        # Compute average episodic reward for this batch
        ep_rewards = [sum(ep_rews) for ep_rews in batch_rews_all]
        avg_reward = sum(ep_rewards) / len(ep_rewards)

        return avg_reward

                
    def evaluate(self,batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        # Squeeze tensors into a single array instead of multiple arrays in an array
        # batch_obs has the shape (timesteps_per_batch, obs_dim), and we only want timesteps_per_batch, therefore we squeeze
        V = self.critic(batch_obs).squeeze(-1) 
        
        # Calculate log_prob of batch actions using most recent actor network
        # Query the actor network for raw logits (non-softmaxed)
        logits = self.actor(batch_obs)
        
        # Categorical Distribution
        dist = Categorical(logits=logits)
        
        # Get log probability of batch actions
        log_probs = dist.log_prob(batch_acts)
        
        # Get entropy of distrubution
        entropy = dist.entropy() # shape: (batch_size,)
        
        # Return predicted V values and log_probs
        return V, log_probs, entropy
    
    def rollout(self):
        # Batch data
        batch_obs = []              # batch observations
        batch_acts = []             # batch actions
        batch_log_probs = []        # log probs of each action
        
        batch_rews = []             # batch rewards
        batch_lens = []             # episodic lengths in batch
        batch_vals = []             # batch critic values
        batch_dones = []            # Done flags
        
        # Number of timesteps run so far this batch    
        t = 0
    
        while t < self.timesteps_per_batch:
            # Episodic data
            ep_rews = []                # episodic rewards
            ep_vals = []                # episodic critic values
            ep_dones = []               # episodic done flags
            
            robot_controller = RobotController(None, self.player)
            factory_controller = FactoryController(None, self.player)
            
            obs, _ = self.env.reset()
            done = False
            
            # Copy observations to maintain a dict version for game_state creation
            obs_dict = copy.deepcopy(obs)
            
            # Parse obs to numpy format for storage
            obs_np = obs_parser(obs_dict, self.env)
            
            # Convert numpy obs to torch tensor for critic
            obs = torch.tensor(obs_np, dtype=torch.float)
            
            # Value cache for macro actions to execute correctly
            remaining_macro_action_queue_length = 0   # initially empty
            macro_action = None             # action (index format)
            macro_log_prob = None           # action log prob
            macro_critic_val = None         # critic value for value loss
            macro_obs_np = None             # observations at the time of macro action selection
            macro_reward = 0                # reward collected during macro action
            
            for ep_t in range(self.max_timesteps_per_episode):
                # If episode is done, break
                if done:
                    break
                
                # collect done value
                ep_dones.append(done)
                
                # Increment timesteps for this batch
                t += 1

                # If the unit has no action in its action queue
                if remaining_macro_action_queue_length <= 0:
                    # Get action from Categorical Distribution sampling (Softmax distribution) along with its log_probs and converted lux_action dict
                    action, log_prob, lux_action_dict, macro_action_length = self.get_action(obs, obs_dict, robot_controller, factory_controller)

                    remaining_macro_action_queue_length = macro_action_length - 1 # subtract for the upcoming step
                    macro_action = action # store action index in cache
                    macro_log_prob = log_prob # Store log_prob for action in cache
                    macro_critic_val = self.critic(obs).detach().flatten() # Poll critic network to value loss calc
                    macro_obs_np = obs_np # Store numpy observations in cache
                    macro_rewards = [] # reset raw per-step reward for macro action
                else:
                    remaining_macro_action_queue_length -= 1 # Reduce action queue counter accordingly
                    # Poll a new action set only for factories
                    lux_action_dict = factory_action_parser(self.env, obs_dict, factory_controller) 
                    
                
                obs, rew, terminated, truncated, _ = self.env.step(lux_action_dict, obs, self.env)
                done = terminated or truncated
                
                done = done["player_0"]
                
                macro_rewards.append(rew) # store macro reward for each step
                
                # Store new observation dict for conversion on next loop
                obs_dict = copy.deepcopy(obs)
                
                # Parse obs to numpy format for storage
                obs_np = obs_parser(obs_dict, self.env)
                
                # Convert numpy obs to torch tensor for critic
                obs = torch.tensor(obs_np, dtype=torch.float)

                if remaining_macro_action_queue_length <= 0 or done:
                    # Discount macro action rewards to properly match discounted rewards
                    macro_reward = sum(self.gamma ** i * r for i, r in enumerate(macro_rewards))
                    
                    # Collect obs, reward, action and log prob
                    batch_obs.append(macro_obs_np)
                    batch_acts.append(macro_action)
                    batch_log_probs.append(macro_log_prob)
                    ep_rews.append(macro_reward)
                    ep_vals.append(macro_critic_val)
                
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # Increment because timestep starts at 0
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            
        # Reshape data as tensors before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
                
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
    
    
    def get_action(self, obs, obs_dict, robot_controller, factory_controller):
        # Query actor network for action
        logits = self.actor(obs)
        
        # Create Categorical distribution for action sampling
        # Handles softmax internally
        dist = Categorical(logits=logits)
        
        # Sample action
        action_index = dist.sample()
        
        # Get log prob of sampled action
        log_prob = dist.log_prob(action_index)
        
        # Create action dict to pass into Lux
        lux_action_dict, macro_action_length = parse_all_actions(self.env, obs_dict, action_index.item(), self.role, robot_controller, factory_controller)
        
        # possibly set in 1st round check, as no units exists yet to avoid skewed data
        
        # Return action, log prob and lux action
        # log_prob is allowed to remain a tensor as we need the graph
        return action_index, log_prob.detach(), lux_action_dict, macro_action_length
    
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = [] # List to store computed advantages for each timestep
        
        # Iterate over each episode's rewards, values, and done flags
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = [] # List to store advantages for the current episode
            last_advantage = 0 # Initialize the last computed advantage
            
            # Calculate episode advantage in reverse order (from last timestep to first)
            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    # Calculate the temporal difference (TD) error for the current timestep
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    # Special case at the boundary (last timestep)
                    delta = ep_rews[t] - ep_vals[t]
                    
                    # Calculate Generalized Advantage Estimation (GAE) for the current timestep
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage # Update the last advantage for the next timestep
                advantages.insert(0, advantage) # Insert advantage at the beginning of the list
                
                # Extend the batch_advantages list with advantages computed for the current episode
            batch_advantages.extend(advantages)
        
        # Convert the batch_advantages list to a PyTorch tensor of type float
        return torch.tensor(batch_advantages, dtype=torch.float)
    
    # Save function for the model
    def save(self, path="rbppo_checkpoint.pth"):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'role': self.role
        }, path)
        print(f"Model saved to {path}")

    # Load function for the model
    def load(self, path="rbppo_checkpoint.pth"):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return False
        data = torch.load(path)
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.actor_optim.load_state_dict(data['actor_optim'])
        self.critic_optim.load_state_dict(data['critic_optim'])
        self.role = data.get('role', self.role)
        print(f"Model loaded from {path}")
        return True


# PPO Training code
reward_log_path = "training_rewards.json"
reward_history = []

# Load previous reward log if available
if os.path.exists(reward_log_path):
    with open(reward_log_path, "r") as f:
        reward_history = json.load(f)

count = 0
env = LuxCustomEnv()
model = PPO(env)

try:
    model.load("rbppo_checkpoint.pth")
except FileNotFoundError:
    print("No checkpoint found, starting fresh.")

while count < 10:
    avg_reward = model.learn(10000)
    reward_history.append(avg_reward)

    # Save model and reward history
    model.save("rbppo_checkpoint.pth")
    with open(reward_log_path, "w") as f:
        json.dump(reward_history, f)
        
    count += 1