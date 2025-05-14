# Obs format (15 obs):
# [
    # 0. unit_pos_x, 
    # 1. unit_pos_y, 
    # 2. unit_type, 
    # 3. unit_power, 
    # 4. unit_ice, 
    # 5. unit_ore, 
    # 6. unit_team_id, 
    # 7. closest_factory_distance_value, 
    # 8. factory_ice, 
    # 9. factory_ore, 
    # 10. closest_ice_scalar_value, 
    # 11. closest_ore_scalar_value
# ]

# Creation steps:
# 1: Understand normalization of obs for proper utilization (done)
# 2: Setup variable for reward shaping strength (e.g. 1 = light, 2 = heavy) (done)
# 3: Reward unit for cargo increase in either robot or factory (light) (done)
# 4: Reward unit for moving towards resource if cargo is empty (heavy) (done)
# 5: Reward unit for moving towards factory if cargo is >0 (heavy)(done)
# 6: Consider static or dynamic role, and if all rewards are equal with role boosting role-specific actions (ice as ice-miner and etc.)
# 7: Consider macro reward setup later
# 8: Dont forget 1000-turn survival reward (consider ice added for each ore transferred to factory, to keep training fair for ore miner)

def reward_parser(prev_obs, new_obs, heavy_shaping=False):
    reward = 0
    
    # Reward thresholds (decimals representing %)
    power_recharge_reward_threshold = 0.1
    power_pickup_reward_threshold = 0.2
    unit_total_cargo_threshold = 0.2
    
    ## --- pre-step data --- ##
    # unit pos (note: normalized pos)
    prev_unit_x = prev_obs[0]
    prev_unit_y = prev_obs[1]
    
    # unit cargo
    prev_unit_power = prev_obs[3]
    prev_unit_ice_cargo = prev_obs[4]
    prev_unit_ore_cargo = prev_obs[5]
    prev_unit_combined_cargo = prev_unit_ice_cargo + prev_unit_ore_cargo
    
    # distance to closest factory
    prev_factory_distance = prev_obs[7]
    
    # factory cargo
    prev_factory_ice_cargo = prev_obs[8]
    prev_factory_ore_cargo = prev_obs[9]
    prev_factory_combined_cargo = prev_factory_ice_cargo + prev_factory_ore_cargo
    
    # distance to closest resource
    prev_ice_distance = prev_obs[10]
    prev_ore_distance = prev_obs[11]
    
    ## --- end pre-step data --- ##
    
    ## --- post-step data --- ##
    # unit pos (note: normalized pos)
    new_unit_x = new_obs[0]
    new_unit_y = new_obs[1]
    
    # unit cargo
    new_unit_power = new_obs[3]
    new_unit_ice_cargo = new_obs[4]
    new_unit_ore_cargo = new_obs[5]
    new_unit_combined_cargo = new_unit_ice_cargo + new_unit_ore_cargo
    
    # distance to closest factory
    new_factory_distance = new_obs[7]
    
    # factory cargo
    new_factory_ice_cargo = new_obs[8]
    new_factory_ore_cargo = new_obs[9]
    new_factory_combined_cargo = new_factory_ice_cargo + new_factory_ore_cargo
    
    # distance to closest resource
    new_ice_distance = new_obs[10]
    new_ore_distance = new_obs[11]
    
    ## --- end post-step data --- ##
    
    # Recharge or pickup reward (seperate mayb?)
    if (prev_unit_power < new_unit_power) and (prev_unit_power < power_recharge_reward_threshold):
        reward += 1
    
    # Dig reward - only rewarded if our old cargo is not above certain threshold
    # Possible modification: if dynamic roles, reward more for matching role resource
    if prev_unit_combined_cargo < unit_total_cargo_threshold:
        if (prev_unit_ice_cargo < new_unit_ice_cargo) or (prev_unit_ore_cargo < new_unit_ore_cargo):
            reward += 5
    
    # Transfer reward
    # Possible modification: if dynamic roles, reward more for matching role resource
    if (prev_factory_combined_cargo < new_factory_combined_cargo):
        reward += 5
    
    if heavy_shaping:
        # Reward for moving closer to resource tile
        # Possible modification: add role to determine choice as it might move away here for small rewards
        # Possible optimization: swap statements to return early
        if (prev_ice_distance > new_ice_distance) or (prev_ore_distance > new_ore_distance):
            # Only reward if old cargo is below threshold (possible mod: empty?)
            if prev_unit_combined_cargo < unit_total_cargo_threshold:
                reward += 2
        
        # Reward for going home with something in cargo
        if (prev_factory_distance > new_factory_distance) and (new_unit_combined_cargo > 0):
            reward += 2
    
    return reward

# note to self: add factory power to obs and split power reward into 2 - 1 for recharge and 1 for pickup