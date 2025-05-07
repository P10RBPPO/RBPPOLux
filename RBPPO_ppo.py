import torch
import numpy as np
import copy
import gymnasium as gym

from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import torch.nn.functional as F

from RBPPO_lux_obs_parser import obs_parser
from RBPPO_network import FeedForwardNN
from RBPPO_lux_env import LuxCustomEnv

from lux.kit import obs_to_game_state, GameState
from RBPPO_lux_action_parser import parse_actions

class PPO:
    def __init__(self, env):
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
        
        # Init Optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
    def _init_hyperparameters(self):
        #  Default values for now
        self.timesteps_per_batch = 1000         # Timesteps per batch
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
        
        self.role = self.roles[0]               # Desired role for training
    
    def learn(self, total_timesteps):
        t_so_far = 0 # Timestep counter
        
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
            
            # Calculate advantage using GAE
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones) 
            V = self.critic(batch_obs).squeeze()
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
                    critic_loss = nn.MSELoss()(V, mini_rtgs)
                
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
                
                # Loss print
                # avg_loss = sum(loss) / len(loss)
                # print(avg_loss)
                
    def evaluate(self,batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs
        # Squeeze tensors into a single array instead of multiple arrays in an array
        # batch_obs has the shape (timesteps_per_batch, obs_dim), and we only want timesteps_per_batch, therefore we squeeze
        V = self.critic(batch_obs).squeeze() 
        
        # Calculate log_prob of batch actions using most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        
        # Return predicted V values and log_probs
        return V, log_probs, dist.entropy()
    
    def rollout(self):
        # Batch data
        batch_obs = []              # batch observations
        batch_acts = []             # batch actions
        batch_log_probs = []        # log probs of each action
        batch_rews = []             # batch rewards
        #batch_rtgs = []            # batch rewards-to-go
        batch_lens = []             # episodic lengths in batch
        
        batch_vals = []             # batch critic values
        batch_dones = []            # Done flags
        
        # Episodic data
        ep_rews = []                # episodic rewards
        ep_vals = []                # episodic critic values
        ep_dones = []               # episodic done flags
        
        # Number of timesteps run so far this batch    
        t = 0
    
        while t < self.timesteps_per_batch:
            # Rewards this ep
            ep_rews = []
            ep_vals = []
            ep_dones = []
            
            obs, _ = self.env.reset()
            done = False
            
            obs_dict = copy.deepcopy(obs)
            
            for ep_t in range(self.max_timesteps_per_episode):
                ep_dones.append(done)
                
                # Increment timesteps for this batch
                t += 1
                
                # Parse obs to numpy format
                obs = obs_parser(obs_dict, self.env)
                
                # Collect obs
                batch_obs.append(obs)
                
                # Convert numpy obs to torch tensor
                obs = torch.tensor(obs, dtype=torch.float)
                
                # Get action from Softmax distribution of actions and output action, lux action and log prob
                action, log_prob, lux_action_dict = self.get_action(obs, obs_dict)
                val = self.critic(obs) 
                
                obs, rew, terminated, truncated, _ = self.env.step(lux_action_dict)
                done = terminated or truncated
                
                # Store new observation dict for conversion on next loop
                obs_dict = copy.deepcopy(obs)
                
                # Collect reward, action and log prob
                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                # If episode is done, break
                if done["player_0"]:
                    break
                
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # Increment because timestep starts at 0
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            
        # Reshape data as tensors before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        #batch_rtgs = self.compute_rtgs(batch_rews)
        
        return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
    
    
    def get_action(self, obs, obs_dict):
        # Query actor network for action (equal to self.actor.forward(obs))
        action = self.actor(obs)
        
        # Log Softmax our output for log probs
        log_softmax = torch.nn.LogSoftmax(dim=0)
        log_probs = log_softmax(action)
        
        # Convert to numpy before action selection
        action = action.detach().numpy()
        
        # Create action dict to pass into Lux and return tensor for chosen action for PPO evaluation
        lux_action_dict, output_action, log_prob_index = parse_actions(self.env, obs_dict, action, self.role)
        
        log_prob = log_probs[log_prob_index]
        
        # Return action, log prob and lux action
        # detach().numpy() to convert from tensor to numpy array
        # log_prob is allowed to remain a tensor as we need the graph
        return output_action, log_prob.detach(), lux_action_dict
    
    
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

# PPO Test code
env = LuxCustomEnv()
model = PPO(env)
model.learn(2000)