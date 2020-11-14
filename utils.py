import numpy as np

def shortest_distance(base_pos, agents_pos):
    distance = float("+inf")
    for agent_pos in agents_pos:
        norm = np.sqrt(np.sum((agent_pos - base_pos) ** 2))
        if norm < distance:
            distance = norm

    return distance

def is_solved(bases_pos, agents_pos, tolerance=0.1):

    for base_pos in bases_pos:
        t = shortest_distance(base_pos, agents_pos)
        if t > tolerance:
            return False

    return True

def is_solved_save(bases_pos, agents_pos, tolerance=0.1):
    t = []

    for base_pos in bases_pos:
        t.append(shortest_distance(base_pos, agents_pos))

    if min(t) <= tolerance:
        return True
    return False

def get_reward(bases_pos, agents_pos):

    reward = 0

    for base_pos in bases_pos:
        reward -= shortest_distance(base_pos, agents_pos)

    return reward


def alter_state(s, bases_pos):
    s_altered = np.array(s)

    for i in range(len(s)):
        current_pos = s_altered[i][2:4]
        entities = np.array(bases_pos).flatten() - np.tile(current_pos, 3).flatten()
        s_altered[i][4:10] = entities

    return list(s_altered)
