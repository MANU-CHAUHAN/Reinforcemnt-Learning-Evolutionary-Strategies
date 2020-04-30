import numpy as np
import random
import time
import gym


# Returns the total reward for given env, policy for episode_len times
def run_episode(env, policy, episode_len=100):
    total_reward = 0
    observation = env.reset()
    for _ in range(episode_len):
        env.render()
        action = policy[observation]  # get best action using the given from current state
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


# Evaluates the policy passed within the provided environment by collecting and averaging rewards from each episode
def evaluate_policy(env, policy, n_episodes=100):
    total_reward = 0.0
    for _ in range(n_episodes):
        total_reward += run_episode(env, policy)
    return total_reward / n_episodes


# Used for returning random policy having range values from 0-3 and size=16
def get_random_policy():
    return np.random.choice(a=4, size=16)


# Implements Crossover with parents (mates) to get children
def crossover(policy1, policy2):
    new_policy = policy1.copy()
    for _ in range(16):
        if np.random.uniform() > 0.5:
            new_policy[_] = policy2[_]
    return new_policy


# Implements mutation to slightly vary the off spring's traits
def mutation(policy, p=0.05):
    new_policy = policy.copy()
    for _ in range(16):
        if np.random.uniform() < p:
            new_policy[_] = np.random.choice(4)
    return new_policy


if __name__ == '__main__':
    random.seed(101)
    np.random.seed(101)

    env = gym.make('FrozenLake-v0') # Solving FrozenLake environment here
    env.seed(0)
    n_policy = 100
    n_steps = 20
    start = time.time()
    policy_population = [get_random_policy() for _ in range(100)]

    for num in range(n_steps):
        policy_scores = [evaluate_policy(env, policy) for policy in policy_population]
        print("\nGeneration number:%d  Policy with max score %0.5f%d".format((num + 1), max(policy_scores)))
        policy_ranks = list(reversed(np.argsort(policy_scores)))
        top_5_policies = [policy_population[x] for x in policy_ranks[:5]]
        select_probs = np.array(policy_scores) / np.sum(policy_scores)
        child_set = [crossover(policy_population[np.random.choice(range(n_policy), p=select_probs)],
                               policy_population[np.random.choice(range(n_policy), p=select_probs)]) for _ in
                     range(0, n_policy - 5)]

        mutated = [mutation(p) for p in child_set]

        policy_population = top_5_policies + mutated

        policy_scores = [evaluate_policy(env, policy) for policy in policy_population]

        fittest_policy = policy_population[np.argmax(policy_scores)]

        end = time.time()

        print("\nBest policy found out in time: %f  having score:%f".format((end - start), fittest_policy))

        print("\n\n\n\nEvaluation\n")
        for _ in range(200):
            run_episode(env, fittest_policy)
        env.close()

