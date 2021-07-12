import numpy as np
import gym
#from gym import wrappers

#https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa


def value_iteration(env, GAMMA=1.0):
    """
    Value-iteration algorithm: Makes a table of how good each state action pair is
    """
    v = np.zeros(env.nS) # initialize value funcion, env.nS is number of states
    MAX_ITERATIONS = 100000
    EPS = 1e-20
    for i in range(MAX_ITERATIONS):
        prev_v = np.copy(v) # remember the previous set of v
        for s in range(env.nS): # for each state in the list of states
            #q_sa = np.zeros(env.action_space.n)
            #q_sx = [sum([p*(r+prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)]
            #print("q_sx: ", q_sx)
            q_sa=[]
            # for each possible a, generate an array q_sa using the value iteration equation
            for a in range(env.nA):
                #env.P[s][a] gives all possible moves, with probability p, s', reward, and if game is over (t/f)
                    q_sa.append(sum(p*(r+prev_v[s_]) for p, s_, r, _ in env.P[s][a]))
                    
            v[s] = np.max(q_sa) # the best state value is the max q_as
            #print("v[s] = ", v[s])

        # check to see if the v tables have converged 
        print(np.sum(np.fabs(prev_v - v)))
        if (np.sum(np.fabs(prev_v - v)) <= EPS):
            print('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v


def extract_policy(v, GAMMA=1.0):
    """Extract a policy given a value-function, input of v from value_iteraciton def"""
    policy=np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n) # create q value array of zeros size of action space
        for a in range(env.action_space.n): # for each action in the possible action spaces
            for next_Sr in env.P[s][a]:
                p, s_, r, _= next_Sr
                q_sa[a] += (p * (r+GAMMA*v[s_]))
        policy[s] = np.argmax(q_sa)
        print("policy[s]:", policy[s])

    return policy


def evaluate_policy(env,policy,GAMMA=1.0, N=100):
    """ Evaluates a policy by running it n times
    returns:
    average total reward
    """
    scores = [run_episode(env, policy, GAMMA=GAMMA, render=False) for _ in range(N)]
    return np.mean(scores) # returns the average of the scores


def run_episode(env, policy, GAMMA=1.0, render=False):
    """ Evaluates a policy by using it to run an episode and finding its total reward
    
    args: 
    env - gym environment
    policy - the policy to be used
    gamma - discount factor
    render - boolean to turn rendering on/off

    returns:
    total reward - real value of the total reward recieved by agent under policy
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(int(policy[obs]))
        total_reward += (GAMMA ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward



if __name__ == '__main__':
    #ENV_NAME = 'FrozenLake8x8-v0'
    ENV_NAME = 'FrozenLake-v0'
    GAMMA = 1.0

    env = gym.make(ENV_NAME)
    #print(env.nA) -> 4
    #print(env.action_space.n) -> 4

    # First find the optimal value fuction
    optimal_v = value_iteration(env,GAMMA)

    # Use the optimal value function to find the optimal policy
    policy = extract_policy(optimal_v, GAMMA)

    # check how good the policy is
    policy_score = evaluate_policy(env,policy, GAMMA, N=1000)
    print('Policy average score = ', policy_score)
