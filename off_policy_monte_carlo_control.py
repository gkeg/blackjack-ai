import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    EPS = 0.05
    # No need to discount future rewards; rewards are certain therefore GAMMA = 1
    GAMMA = 1.0

    # Define the spaces of our environment and agent
    agent_sum_space = [i for i in range(4, 22)]
    dealer_show_card_space = [i+1 for i in range(10)]
    agent_ace_space = [False, True]
    action_space = [0, 1]
    state_space = []

    # Agent's estimate of the expected future reward
    Q = {}
    # C is the sum of the relative weights of the above trajectories occuring
    # under both the target and behaviour policy
    C = {}

    # Initialize to 0 for every state in our space
    for total in agent_sum_space:
        for card in dealer_show_card_space:
            for ace in agent_ace_space:
                for action in action_space:
                    Q[((total, card, ace), action)] = 0
                    C[((total, card, ace), action)] = 0
                state_space.append((total, card, ace))

    # What we're using to calculate the optimal policy, argmax of the agent's
    # estimate of discounted future rewards
    # Numpy argmax always will choose stick if equal, we want to alternate between
    # stick and hit
    target_policy = {}
    for state in state_space:
        values = np.array([Q[(state, a)] for a in action_space])
        # Get the argmax
        best = np.random.choice(np.where(values == values.max())[0])
        target_policy[state] = action_space[best]

    # Play 30000 games
    num_episodes = 30000
    for i in range(num_episodes):
        memory = []
        if i % 1000 == 0:
            print('Starting episode ' + str(i))

        behaviour_policy = {}
        for state in state_space:
            rand = np.random.random()
            if rand < 1 - EPS:
                # Greedy action
                behaviour_policy[state] = [target_policy[state]]
            else:
                # Random action in the action space
                behaviour_policy[state] = action_space

        observation = env.reset()
        done = False

        # Play the game until completion
        while not done:
            action = np.random.choice(behaviour_policy[observation])
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], observation[2], action, reward))
            observation = observation_

        # Make sure we get the last observation as well
        memory.append((observation[0], observation[1], observation[2], action, reward))

        # Now, iterate backwards over the memory
        G = 0
        W = 1
        last = True
        for player_sum, dealer_card, usable_ace, action, reward in reversed(memory):
            sa = ((player_sum, dealer_card, usable_ace), action)

            # Skip over the last entry
            if last:
                last = False
            else:
                C[sa] += W
                Q[sa] += (W / C[sa]) * (G - Q[sa])
                values = np.array([Q[(state, a)] for a in action_space])
                best = np.random.choice(np.where(values == values.max())[0])
                target_policy[state] = action_space[best]

                if action != target_policy[state]:
                    break
                if len(behaviour_policy[state]) == 1:
                    prob = 1 - EPS
                else:
                    prob = EPS / len(behaviour_policy[state])
                W *= 1/prob
                G = GAMMA*G + reward

        # Update epsilon
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0

    # Play some games with our trained model; our training sequence
    num_episodes = 1000
    rewards = np.zeros(num_episodes)
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    print('Getting ready to test target policy')
    for i in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = target_policy[observation]
            observation_, reward, done, info = env.step(action)
            observation = observation_
        total_reward += reward
        rewards[i] = total_reward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws += 1
        elif reward == -1:
            losses += 1

    wins /= num_episodes
    losses /= num_episodes
    draws /= num_episodes

    print('Win rate: ' + str(wins) + ' loss rate: ' + str(losses) + ' draw rate: ' + str(draws))
    plt.plot(rewards)
    plt.show()



