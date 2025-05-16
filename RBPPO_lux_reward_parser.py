# Obs format (12 obs):
# [
    # 0. unit_pos_x, 
    # 1. unit_pos_y, 
    # 2. unit_type, 
    # 3. unit_power, 
    # 4. unit_ice, 
    # 5. unit_ore, 
    # 6. unit_team_id, 
    # 7. closest_factory_distance_value, 
    # 8. factory_power
    # 9. factory_ice, 
    # 10. factory_ore, 
    # 11. closest_ice_scalar_value, 
    # 12. closest_ore_scalar_value
# ]

# Possible Modifications:
# 1: Reward for winning / Penalty for losing?

def reward_parser(prev_obs, new_obs, role, heavy_shaping=False):
    reward = 0
    
    # Reward thresholds (decimals representing %)
    power_recharge_reward_threshold = 0.1
    power_pickup_reward_threshold = 0.25
    unit_total_cargo_reward_reduction_threshold = 0.1
    unit_total_cargo_reward_cap = 0.2
    
    ## --- pre-step data --- ##
    # unit pos (note: normalized pos)
    #prev_unit_x = prev_obs[0]
    #prev_unit_y = prev_obs[1]
    
    # unit cargo
    prev_unit_power = prev_obs[3]
    prev_unit_ice_cargo = prev_obs[4]
    prev_unit_ore_cargo = prev_obs[5]
    prev_unit_combined_cargo = prev_unit_ice_cargo + prev_unit_ore_cargo
    
    # distance to closest factory
    prev_factory_distance = prev_obs[7]
    
    # factory cargo
    prev_factory_power = prev_obs[8]
    prev_factory_ice_cargo = prev_obs[9]
    prev_factory_ore_cargo = prev_obs[10]
    prev_factory_combined_cargo = prev_factory_ice_cargo + prev_factory_ore_cargo
    
    # distance to closest resource
    prev_ice_distance = prev_obs[11]
    prev_ore_distance = prev_obs[12]
    
    ## --- end pre-step data --- ##
    
    ## --- post-step data --- ##
    # unit pos (note: normalized pos)
    #new_unit_x = new_obs[0]
    #new_unit_y = new_obs[1]
    
    # unit cargo
    new_unit_power = new_obs[3]
    new_unit_ice_cargo = new_obs[4]
    new_unit_ore_cargo = new_obs[5]
    new_unit_combined_cargo = new_unit_ice_cargo + new_unit_ore_cargo
    
    # distance to closest factory
    new_factory_distance = new_obs[7]
    
    # factory cargo
    new_factory_power = prev_obs[8]
    new_factory_ice_cargo = new_obs[9]
    new_factory_ore_cargo = new_obs[10]
    new_factory_combined_cargo = new_factory_ice_cargo + new_factory_ore_cargo
    
    # distance to closest resource
    new_ice_distance = new_obs[11]
    new_ore_distance = new_obs[12]
    
    ## --- end post-step data --- ##
    
    # Recharge reward
    if  prev_unit_power < power_recharge_reward_threshold:
        if (prev_unit_power < new_unit_power) and (prev_factory_power < new_factory_power):
            reward += 1
    
    # Pickup reward
    if  prev_unit_power < power_pickup_reward_threshold:
        if (prev_unit_power < new_unit_power) and (prev_factory_power > new_factory_power):
            reward += 2
    
    # Dig reward
    if prev_unit_combined_cargo < unit_total_cargo_reward_cap:
        if (prev_unit_ice_cargo < new_unit_ice_cargo) or (prev_unit_ore_cargo < new_unit_ore_cargo):
            if prev_unit_combined_cargo < unit_total_cargo_reward_reduction_threshold:
                if (prev_unit_ice_cargo < new_unit_ice_cargo) and role == "Ice Miner":
                    reward += 10
                elif (prev_unit_ore_cargo < new_unit_ore_cargo) and role == "Ore Miner":
                    reward += 10
                else:
                    reward += 5
            else:
                if (prev_unit_ice_cargo < new_unit_ice_cargo) and role == "Ice Miner":
                    reward += 5
                elif (prev_unit_ore_cargo < new_unit_ore_cargo) and role == "Ore Miner":
                    reward += 5
                else:
                    reward += 2
        
    # Transfer reward
    if prev_factory_combined_cargo < new_factory_combined_cargo:
        if (prev_factory_ice_cargo < new_factory_ice_cargo) and role == "Ice Miner":
            reward += 10
        elif (prev_factory_ore_cargo < new_factory_ore_cargo) and role == "Ore Miner":
            reward += 10
        else:
            reward += 5
    
    # Heavy shaping - Rewards for moving closer to resources and going home with resources in cargo
    if heavy_shaping:
        
        # Reward for moving closer to resource tile
        if prev_unit_combined_cargo < unit_total_cargo_reward_cap:
            if (prev_ice_distance > new_ice_distance) or (prev_ore_distance > new_ore_distance):
                if (prev_ice_distance > new_ice_distance) and role == "Ice Miner":
                    reward += 4
                elif (prev_ore_distance > new_ore_distance) and role == "Ore Miner":
                    reward += 4
                else:
                    reward += 2
        
        
        # Reward for going home with something in cargo
        if (prev_factory_distance > new_factory_distance) and (new_unit_combined_cargo > 0):
            if (new_unit_ice_cargo > 0) and role == "Ice Miner":
                reward += 6
            elif(new_unit_ore_cargo > 0) and role == "Ore Miner":
                reward += 6
            else:
                reward += 3
    
    return reward