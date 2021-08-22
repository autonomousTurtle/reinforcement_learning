import numpy as np
import gym
from numpy.random.mtrand import rand

env=gym.make('FrozenLake-v0')

env.reset()
env.render()


#Parameters
MAX_ITERATIONS = 10

#print("Action space: ", env.action_space)
# Discrete(4)
#print("Observation space: ", env.observation_space)
# Discrete(16)

# value function iteration
def value_iteration(env, MAX_ITERATIONS, LMBDA=0.9):
    stateValue = [0 for i in range(env.nS)]
    newStateValue = stateValue.copy()
    for i in range(MAX_ITERATIONS):
        for state in range(env.nS):
            action_values = []
            for action in range(env.nA):
                state_value = 0
                for i in range(len(env.P[state][action])):
                    prob, next_state, reward, done = env.P[state][action][i]
                    state_action_value = prob * (reward+LMBDA*stateValue[next_state])
                    state_value += state_action_value
                action_values.append(state_value) # the value of each action
                best_action = np.argmax(np.asarray(action_values)) # choose the action which gives the max value
                newStateValue[state] = action_values[best_action] #update the value of the state with the best action
            
        if i > 1000:
            if sum(stateValue) - sum(newStateValue) < 1e-4: # if there is negligible difference between old and new states
                break
                print(i)    
        else:
            stateVale = newStateValue.copy()
    return stateValue

value_iteration(env, MAX_ITERATIONS)
for i in range(MAX_ITERATIONS):
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    env.render()
    #print(env.nS)
    #print(env.P)
    if done:
        break

