# -*- coding: utf-8 -*-
"""
Created on 2019.04.03

This code is to conudct the experiments to evaluate the performance of algorithm

@author: Administrator
"""

import numpy as np
import scipy.io as sio   # import the state's parameter in each slot
from ddpg import DDPG
from ou_noise import OUNoise
import time

is_batch_norm = True   #batch normalization switch

def main():
    
    '''
    In this file, we first load the system state parameter from the .mat files, then for each 
    each slot, we observe the state parameter and make the action. Then, we save this state-actor
    record into the memory for the latter train. Finally, the system convert into te next ecopids.
    '''
    #load the state parameter form .mat file
    task_size = sio.loadmat('./data/data')['input_data_size']   #load the task size 
    CPU_density = sio.loadmat('./data/data')['input_CPU_density']   #load the required CPU cycles of each task bit 
    task_delay_re = sio.loadmat('./data/data')['input_task_delay_re']  #load the maximum toleration delay of each task
    task_gain = sio.loadmat('./data/data')['input_task_gain']  #load the gain of each task
    user_power = sio.loadmat('./data/data')['input_user_power']  #load the transmit power of each user
    user_chan_gain = sio.loadmat('./data/data')['input_user_chan_gain']  #load the wireless channel gain of each user
    bs_capacity = sio.loadmat('./data/data')['input_bs_capacity']  #load the computing capacity of each base station    
    
    
    #set the number of users in these base station
    bs_1_user_num = 10
    bs_2_user_num = 20
    bs_3_user_num = 10
    
    #set the wireless channel nosie, channel bandiwidth, transmission rate of wired connection,
    chan_noise =   10**(-8)
    chan_band = 10**6
    wired_rate = 10
    
    #set the length of time slot 
    slot_len = 10000
    
    #Set the record number in the replay buffer, the total reward, the reward record of the whole time slots
    counter = 0 
    total_reward = 0
    reward_st = np.array([0])
    
    #Randomly initialize critic,actor,target critic, target actor network and replay buffer
    num_states, num_actions = len(task_size[:,1]) * 7, len(task_size[:,1])
    agent = DDPG(num_states, num_actions, is_batch_norm)
    
    #set the explore nosie to guarantee the algrithm's optimal performance
    exploration_noise = OUNoise(1)
    
    #travel each slot, and make the action decision
    for i in range(slot_len):
        print ("==== Starting episode no:",i,"====","\n")
        current_state = np.hstack((task_size[:,i], CPU_density[:,i], task_delay_re[:,i], task_gain[:,i],\
        user_power[:,0], user_chan_gain[:,i],bs_capacity[:,i]))   #obtain the current system state
        current_state = np.reshape(current_state, [1, -1])
        actor_input = current_state   #set the input of actor network
        actor_output = agent.evaluate_actor(actor_input)   #predict the action in this slot
        noise = exploration_noise.noise()   #obtain the noise added in the action
        action = actor_output[0] + noise #Select action according to current policy and exploration noise
#        print ("Action at slot", i ," :",action,"\n")
        reward = 1#fuction(action,current_state)   #obtain the reward in this slot
        next_state = np.hstack((task_size[:,i+1], CPU_density[:,i+1], task_delay_re[:,i+1], task_gain[:,i+1], user_power[:,0],\
        user_chan_gain[:,i+1], bs_capacity[:,i+1]))   #obtain the system state in the next slot
        next_state = np.reshape(next_state, [1, -1])
        agent.add_experience(current_state, next_state, action, reward)   #add s_t,s_t+1,action,reward to experience memory
        #train critic and actor network
        if counter > 64: 
            agent.train()
        counter+=1
#        print ('EPISODE: ',i,'Reward: ',reward)
        reward_st = np.append(reward_st,reward)
        np.savetxt('episode_reward.txt',reward_st, newline="\n")
    total_reward+=reward
    print ("Average reward per episode {}".format(total_reward / slot_len))
        
        
        
if __name__ == '__main__':
    main()    