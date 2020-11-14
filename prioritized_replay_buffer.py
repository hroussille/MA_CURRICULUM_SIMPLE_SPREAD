import random
import numpy as np
import copy
from SumTree import SumTree

class PrioritizedReplayBuffer:  # stored as ( s, a, r, s_, subs ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, buffer_size):
        self.tree = SumTree(buffer_size)
        self.capacity = buffer_size
        self.size = 0

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def push(self, s, a, r, done, s_t, subs, error):

        s = copy.deepcopy(s)
        a = copy.deepcopy(a)
        r = copy.deepcopy(r)
        done = copy.deepcopy(done)
        s_t = copy.deepcopy(s_t)
        subs = copy.deepcopy(subs)

        self.size = max(self.size + 1, self.capacity)
        p = self._get_priority(error)
        self.tree.add(p, (s, a, r, done, s_t, subs))

    def __len__(self):
        return self.size

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states = []
        actions = []
        rewards = []
        dones = []
        new_states = []
        sub = []

        for exp in batch:
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            dones.append(exp[3])
            new_states.append(exp[4])
            sub.append(exp[5])


        np.array(states)
        np.array(actions)
        np.array(rewards)
        np.array(dones)
        np.array(new_states)
        np.array(sub)

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(dones), np.array(new_states), np.array(sub)) , np.array(idxs), np.array(is_weight)

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
