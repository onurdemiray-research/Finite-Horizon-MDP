import numpy as np


class MDP(object):

    def __init__(self, n_epoch, states, actions, transition_probabilities, rewards, discount_factor, finals):

        """
        % Construction method

        :param n_epoch: Integer
        :param states: List of integers, the last one represents the absorbing state
        :param actions: List of integers
        :param transition_probabilities: 3-D numpy array,  |S|x|A|x|S|
        :param rewards: 2-D numpy array, |S|x|A|
        :param discount_factor: Float
        :param finals: 1-D numpy array,|S|
        """

        self.T = n_epoch
        self.S = states
        self.A = actions
        self.P = transition_probabilities
        self.R = rewards
        self.gamma = discount_factor
        self.F = finals

        self.V = np.empty(shape=(self.T, len(self.S)))  # 2D Numpy Array
        self.pi = {}  # dictionary, ((time, state), action)

    def backward_induction(self):

        self.V[-1, :] = self.F

        for t in range(self.T - 2, -1, -1):  # backward

            # Compute value functions
            for s in self.S:
                if s != self.S[-1]:
                    self.V[t, s] = max([self.R[s, a] + self.gamma *
                                        np.sum(self.P[s, a, :] * self.V[t + 1, :]) for a in self.A])
                else:
                    self.V[t, s] = self.F[s]

            # Compute Policies
            for s in self.S:
                if s != self.S[-1]:
                    self.pi[t, s] = np.argmax(np.array(
                        [self.R[s, a] + self.gamma *
                         np.sum(self.P[s, a, :] * self.V[t + 1, :]) for a in self.A]
                    ))

    def plot(self, state):

        policy_actions = [self.pi[(t, state)] for t in range(self.T - 1)]