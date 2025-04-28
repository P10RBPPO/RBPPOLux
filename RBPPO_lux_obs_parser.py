import numpy as np

def torch_obs_parser(obs):
    obs_dict = obs
    #print(obs_dict)
    
    key_list = obs_dict.keys()
    
    for key in key_list:
        val = obs_dict.get(key)
        # print(key)
        # print(val)
