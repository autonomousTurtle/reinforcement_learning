import numpy as np
import gym


def policy_iteration(env,GAMMA=1.0):
    """policy iteration algorithm"""
    policy = np.random.choice(env.nA, size=(env.nS)) #initialize a random policy
    MAX_ITERATIONS = 200000
    GAMMA = 1.0
    for i in range(MAX_ITERATIONS):
        old_policy_v = compute_policy_v(env, policy, GAMMA)
        new_policy = extract_policy(env, old_policy_v, GAMMA)
        print(policy-new_policy)
        if (np.all(policy == new_policy)):
            print('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return policy

def compute_policy_v(env, policy, GAMMA = 1.0):
    """Iterativley evaluate the value-function under policy. 
    Alternatively, we could formulate a set of linear equations in terms of v[s]
    and solve them to find the value function"""
    v = np.zeros(env.nS)
    EPS = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p*(r+GAMMA*prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v-v))) <= EPS):
            break
    return v

def extract_policy(env, v, GAMMA=1.0):
    """Extract the policy given a value function"""
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p*(r+GAMMA*v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def evaluate_policy(env, policy, GAMMA=1.0, n=1000):
    scores = [run_episode(env, policy, GAMMA, False) for _ in range(n)]
    return np.mean(scores)


def run_episode(env, policy, GAMMA=1.0, render=False):
    """ runs an episode and returns the total reward"""
    obs= env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done, _ = env.step(policy[obs])
        total_reward += ((GAMMA ** step_idx)*reward)
        step_idx +=1
        if done:
            break
    return total_reward



if __name__ == '__main__':
    ENV_NAME = 'FrozenLake8x8-v0'

    env = gym.make(ENV_NAME)
    optimal_policy = policy_iteration(env, GAMMA=1.0)
    scores = evaluate_policy(env, optimal_policy, GAMMA=1.0)
    print('Average scores = ', np.mean(scores))

