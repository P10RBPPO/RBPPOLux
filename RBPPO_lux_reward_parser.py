# Obs format (15 obs):
# [0. unit_pos_x, 1. unit_pos_y, 2. unit_type, 3. unit_power, 4. unit_ice, 5. unit_ore, 6. unit_team_id, 7. closest_factory_distance_value, 8. factory_ice, 9. factory_ore, 10. closest_ice_scalar_value, 11. closest_ore_scalar_value]

# Reward parser input params: prev_obs, new_obs, env?

# Creation steps:
# 1: Understand normalization of obs for proper utilization (done)
# 2: Setup variable for reward shaping strength (e.g. 1 = light, 2 = heavy)
# 3: Reward unit for cargo increase in either robot or factory (light)
# 4: Reward unit for moving towards resource if cargo is empty (heavy)
# 5: Reward unit for moving towards factory if cargo is >0 (heavy)
# 6: Consider static or dynamic role, and if all rewards are equal with role boosting role-specific actions (ice as ice-miner and etc.)
# 7: Consider macro reward setup later
# 8: Dont forget 1000-turn survival reward (consider ice added for each ore transferred to factory, to keep training fair for ore miner)