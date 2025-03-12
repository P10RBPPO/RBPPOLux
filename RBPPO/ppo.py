import torch

from torch.distributions import MultivariateNormal
from network import FeedForwardNN

class PPO:
    def __init__(self, env):
        # Init hyperparams
        self._init_hyperparameters()
        
        # Get environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Init actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        
        # Create the covariance matrix for get_action - fill_value is stdev value
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        
    def _init_hyperparameters(self):
        #  Default values for now
        self.timesteps_per_batch = 4800         # Timesteps per batch
        self.max_timesteps_per_episode = 1600   # Timesteps per episode    
        self.gamma = 0.95
    
    def learn(self, total_timesteps):
        t_so_far = 0 # Timestep counter
        
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
    
    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch
        
        # Number of timesteps run so far this batch    
        t = 0
    
        while t < self.timesteps_per_batch:
            # Rewards this ep
            ep_rews = []
            
            obs = self.env.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps for this batch
                t += 1
                
                # Collect obs
                batch_obs.append(obs)
                
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
                
                # Collect reward, action and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                if done:
                    break
                
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # Increment because timestep starts at 0
            batch_rews.append(ep_rews)
            
        # Reshape data as tensors before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        
        batch_rtgs = self.compute_rtgs(batch_rews)
        
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def get_action(self, obs):
        # Query actor network for mean action (equal to self.actor.forward(obs))
        mean = self.actor(obs)
        
        # Create Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        
        # Sample an action from the dist and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Return sampled action and log prob
        # detach().numpy() to convert from tensor to numpy array
        # log_prob is allowed to remain a tensor as we need the graph
        return action.detach().numpy(), log_prob.detach()
    
    def compute_rtgs(self, batch_rews):
        # Rewards-to-go per episode per batch to return
        # Shape: num timesteps per episode
        batch_rtgs = []
        
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # Discounted reward so far
            
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
                
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        
        return batch_rtgs 