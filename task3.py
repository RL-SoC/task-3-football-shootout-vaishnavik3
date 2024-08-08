import argparse
import numpy as np
import matplotlib.pyplot as plt

# Define constants
GRID_SIZE = 4
GOAL_POSITIONS = [15, 16]
ACTIONS = {
    'move_left': 0, 'move_right': 1, 'move_up': 2, 'move_down': 3,
    'pass': 8, 'shoot': 9
}

# Define opponent policies (replace with actual file paths)
OPPONENT_POLICIES = {
    'greedy': 'path_to_greedy_policy_file',
    'park_the_bus': 'path_to_park_the_bus_policy_file',
    'random': 'path_to_random_policy_file'
}

class FootballMDP:
    def __init__(self, p, q, opponent_policy):
        self.p = p
        self.q = q
        self.opponent_policy = self.load_opponent_policy(opponent_policy)

    def load_opponent_policy(self, policy_file):
        # Load opponent policy from file
        return np.loadtxt(policy_file)

    def transition_function(self, state, action):
        B1, B2, R, ball_possession = state
        new_state = list(state)
        prob = 0

        if action in [ACTIONS['move_left'], ACTIONS['move_right'], ACTIONS['move_up'], ACTIONS['move_down']]:
            # Movement logic
            if ball_possession == 1:
                prob_success = 1 - 2 * self.p
                prob_loss = 2 * self.p
            else:
                prob_success = 1 - self.p
                prob_loss = self.p

            if action == ACTIONS['move_left']:
                new_state[0] -= 1
            elif action == ACTIONS['move_right']:
                new_state[0] += 1
            elif action == ACTIONS['move_up']:
                new_state[0] -= 4
            elif action == ACTIONS['move_down']:
                new_state[0] += 4

            # Check if B1 moves out of bounds
            if new_state[0] < 0 or new_state[0] >= GRID_SIZE ** 2:
                return state, 0  # Game ends

            if np.random.rand() < prob_loss:
                return state, 0  # Possession lost

            prob = prob_success

        elif action == ACTIONS['pass']:
            distance = abs(state[0] % GRID_SIZE - state[1] % GRID_SIZE) + abs(state[0] // GRID_SIZE - state[1] // GRID_SIZE)
            pass_prob = self.q - 0.1 * distance
            if pass_prob < 0:
                return state, 0  # Pass fails

            if np.random.rand() > pass_prob:
                return state, 0  # Pass fails

            new_state[3] = 3 - state[3]  # Switch ball possession

        elif action == ACTIONS['shoot']:
            distance = abs(state[0] % GRID_SIZE - 2)  # Distance from goal
            shoot_prob = self.q - 0.2 * (3 - distance)
            if shoot_prob < 0:
                return state, 0  # Shot fails

            if np.random.rand() > shoot_prob:
                return state, 0  # Shot fails

            return new_state, 1  # Goal scored

        return tuple(new_state), prob

    def reward_function(self, state):
        if state[0] in GOAL_POSITIONS or state[1] in GOAL_POSITIONS:
            return 1  # Goal scored
        return 0  # No goal or game ends

    def value_iteration(self, discount_factor=0.9, theta=1e-5):
        num_states = GRID_SIZE ** 2 * GRID_SIZE ** 2 * GRID_SIZE ** 2 * 2
        V = np.zeros(num_states)
        policy = np.zeros(num_states, dtype=int)

        while True:
            delta = 0
            for state in range(num_states):
                v = V[state]
                max_value = max(self.qsa(state, action, V, discount_factor) for action in range(10))
                V[state] = max_value
                delta = max(delta, np.abs(v - V[state]))

            if delta < theta:
                break

        for state in range(num_states):
            policy[state] = np.argmax([self.qsa(state, action, V, discount_factor) for action in range(10)])

        return policy, V

    def qsa(self, state, action, V, discount_factor):
        new_state, prob = self.transition_function(state, action)
        reward = self.reward_function(new_state)
        return prob * (reward + discount_factor * V[new_state])

    def evaluate_policy(self, policy, start_state):
        state = start_state
        total_reward = 0
        while True:
            action = policy[state]
            new_state, prob = self.transition_function(state, action)
            reward = self.reward_function(new_state)
            total_reward += reward * prob
            if reward == 1 or prob == 0:
                break
            state = new_state
        return total_reward

    def generate_graphs(self):
        ps = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        qs = [0.6, 0.7, 0.8, 0.9, 1.0]
        graph1 = []
        graph2 = []

        start_state = (5, 9, 8, 1)

        for p in ps:
            self.p = p
            policy, _ = self.value_iteration()
            graph1.append(self.evaluate_policy(policy, start_state))

        for q in qs:
            self.q = q
            policy, _ = self.value_iteration()
            graph2.append(self.evaluate_policy(policy, start_state))

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(ps, graph1, marker='o')
        plt.title('Graph 1: Probability vs p (q=0.7)')
        plt.xlabel('p')
        plt.ylabel('Probability of Winning')

        plt.subplot(1, 2, 2)
        plt.plot(qs, graph2, marker='o')
        plt.title('Graph 2: Probability vs q (p=0.3)')
        plt.xlabel('q')
        plt.ylabel('Probability of Winning')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football MDP")
    parser.add_argument('--p', type=float, default=0.3, help='Probability p')
    parser.add_argument('--q', type=float, default=0.7, help='Probability q')
    parser.add_argument('--opponent', type=str, default='greedy', help='Opponent policy')
    args = parser.parse_args()

    mdp = FootballMDP(p=args.p, q=args.q, opponent_policy=OPPONENT_POLICIES[args.opponent])
    policy, _ = mdp.value_iteration()
    mdp.generate_graphs()
