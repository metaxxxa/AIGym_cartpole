from typing import Sequence
import gym
import numpy as np
import math
import random
import copy
env = gym.make('CartPole-v0')

#metaparameters
alpha = 0.3
discount_factor = 0.99
epsilon_param = 0.01
actions = [0, 1]
number_states = 20

#discretization of states
position = np.linspace(-2,2,number_states) #  np.linspace(-4.8,4.8,number_states)
velocity = np.linspace(-2,2,number_states) # np.linspace(-5,5,number_states) #can go to infinite but does not in practice
angle = np.linspace(-0.418,0.418,number_states)
angular_velocity = np.linspace(-2,2,number_states) #  np.linspace(-3,3,number_states)

def discretize(observation,variable):
    discrete_index = math.floor((observation+variable[-1])/(variable[1]-variable[0])) #returns index of the state for given observation
    if  discrete_index > (number_states-1):
        discrete_index = (number_states-1)
    elif discrete_index < 0:
        discrete_index = 0
    return discrete_index

def find_state(observation):
    cart_position = discretize(observation[0], position)
    cart_velocity = discretize(observation[1], velocity)
    pole_angle = discretize(observation[2], angle)
    pole_angular_velocity = discretize(observation[3], angular_velocity)
    return [cart_position, cart_velocity, pole_angle, pole_angular_velocity] #returning index in each finite state property

def add(array, element):
    temp = copy.deepcopy(array)
    temp.append(element)
    return temp

def greedy(state,Q):
    if (Q[tuple(add(state, 0))] != Q[tuple(add(state, 1))]):
        prob = random.random()
        if prob < epsilon_param:
            return random.choice(actions)
        else: 
            if (Q[tuple(add(state, 0))] > Q[tuple(add(state, 1))]):
                return 0
            else:
                return 1
    else:
        return random.choice(actions)

def max_Q(Q, state):
    return max(Q[tuple(add(state, actions[1]))], Q[tuple(add(state, actions[0]))])

#initialize Q
Q = {}
for pos in range(len(position)):
    for vel in range(len(velocity)):
        for ang in range(len(angle)):
            for ang_vel in range(len(angular_velocity)):
                for act in actions:
                    Q[tuple([pos, vel, ang, ang_vel, act ])] = 0

for i_episode in range(10000):
    observation = env.reset() #array 4
    done = False
    for t in range(100):

        env.render()
        action = greedy(find_state(observation),Q)
        previous_observed = find_state(observation)
        observation, reward, done, info = env.step(action)
        Q[tuple(add(previous_observed, action))] += alpha*(reward + discount_factor*max_Q(Q,find_state(observation)) - Q[tuple(add(previous_observed, action))])
        
        if done:
            print("%i th episode finished after %i timesteps"%(i_episode, (t+1)))
            break


print("CART LEARNED")

env.close()
