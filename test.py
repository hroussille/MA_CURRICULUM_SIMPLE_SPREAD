
import numpy as np
import torch
import time
import copy
import utils
from MADDPG import MADDPG
import random
import argparse
import yaml
from SimpleInteractions import SimpleInteractions

from ActorCritic import Actor

env = SimpleInteractions(3, with_finish_zone=False, synchronized_activation=True)

def get_bases_affectations(bases_pos, agents_pos):

    bases_affectations = [None, None, None]

    for index_base, base_pos in enumerate(bases_pos):
        min_dist = float("+inf")

        for index_agent, agent_pos in enumerate(agents_pos):
            distance = np.sqrt(np.sum(np.square(base_pos - agent_pos)))

            if distance <= 0.1 and distance <= min_dist:
                min_dist = distance
                bases_affectations[index_base] = index_agent

    return bases_affectations

def get_learners_subpolicies(learner, env):
    return [random.randrange(0, learner.subpolicies) for _ in range(env.n)]

def load_config(path="config.yaml"):
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config

def is_valid(landmarks):
    for current_index, current_landmark in landmarks:
        for other_index, other_landmark in landmarks:

            if current_index == other_index:
                continue

            if np.linalg.norm(current_landmark - other_landmark) <= 0.1:
                return False

    return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help="path to output folder")

    args = parser.parse_args()
    config = load_config(args.path + "/config.yaml")

    env = SimpleInteractions(3, with_finish_zone=False, synchronized_activation=True)
    learner = MADDPG(env, config['env']['env_obs'], **config['learners'])
    learner.load(args.path + "/models_1/")

    affectations = []

    o = env.reset()
    reward = []

    subs = np.random.randint(0, len(config['env']['env_obs']), config['learners']['subpolicies'])

    average_reward = []

    success_rate = 0
    success_time = []
    total = 0

    while len(affectations) < 1000:


        state = env.reset()

        episode_reward = 0
        subs = get_learners_subpolicies(learner, env)

        for step in range(50):

           # np.random.shuffle(goal_simple)
            #goal = np.tile(goal_simple.flatten(), (env.n, 1))

            with torch.no_grad():
                actors_input = state
                action = learner.act(actors_input, subs, noise=False)

            state, _, done, _ = env.step(action)

            r = int(done)
            episode_reward += r / (step + 1e-5)

            if done:
                #bases_affectations = get_bases_affectations(bases_pos, agents_pos)
                #affectations.append(bases_affectations)
                # print("New affectation : {} ({})".format(bases_affectations, len(affectations)))
                average_reward.append(step)
                success_rate +=1
                success_time.append(step)
                break

        #average_reward.append(episode_reward)

            #env.render(mode="human")
            #time.sleep(0.1)

        total += 1

    print("Success rate : {}".format(success_rate / total))
    print("Average time to success : {}".format(np.mean(success_time)))
    #print(np.mean(average_reward))

    env.close()

    np.save("affectations.npy", np.array(affectations).reshape(-1, 3))
