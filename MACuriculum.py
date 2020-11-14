
from MADDPG import MADDPG
from PPO import PPO
#from PPO_vanilla import PPO
from tqdm import tqdm
from tqdm import trange
from numpy_ringbuffer import RingBuffer

import numpy as np
import random
import torch
import utils
import copy
import time

env_obs = {
           'simple_spread': [14, 14, 14],
           'simple_adversary': [8, 10, 10],
           'simple_tag':[16, 16, 16, 14]
           }

class MACuriculum():

    def __init__(self, env, writer, run_id, config, path):

        self.env = env
        self.config = config
        self.env_name = config['env']['env_name']
        self.total_obs = sum(config['env']['env_obs'])
        self.self_play_gamma = config['self_play']['self_play_gamma']
        self.shuffle_self_play = config['self_play']['shuffle']
        self.shuffle_target_play = config['target_play']['shuffle']
        self.writer = writer
        self.run_id = run_id
        self.path = path

        self.learners = MADDPG(env, config['env']['env_obs'], **config['learners'])
        self.teachers = MADDPG(env, config['env']['env_obs'], **config['teachers'])
        self.stop = PPO(2 * self.total_obs, **config['stop'])

    def get_teachers_subpolicies(self):
        return [random.randrange(0, self.config['teachers']['subpolicies']) for _ in range(self.env.n)]

    def get_learners_subpolicies(self):
        return [random.randrange(0, self.config['learners']['subpolicies']) for _ in range(self.env.n)]

    def apply_noise_decay(self):
        self.learners.apply_noise_decay()
        self.teachers.apply_noise_decay()

    def run(self):
        target_play_mean = []
        target_play_std = []

        self_play_mean = []
        self_play_std = []
        current_best = 0

        max_episodes = self.config['self_play']['episodes']
        max_timestep = self.config['self_play']['max_timestep']
        max_exploration_episodes = self.config['self_play']['exploration_episodes']
        stop_probability = self.config['self_play']['exploration_stop_probability']
        tolerance = self.config['self_play']['tolerance']
        stop_update = self.config['self_play']['stop_update_freq']
        mode = self.config['self_play']['mode']
        alternate = self.config['self_play']['alternate']
        alternate_step = self.config['self_play']['alternate_step']
        test_freq = self.config['self_play']['test_freq']
        test_episodes = self.config['self_play']['test_episodes']
        max_timestep_target = self.config['target_play']['max_timestep']

        max_timestep_strategy = self.config['self_play']['max_timestep_strategy']
        ma_window_length = self.config['self_play']['ma_window_length']
        ma_multiplier = self.config['self_play']['ma_multiplier']
        ma_default_value = self.config['self_play']['ma_default_value']
        ma_bias = self.config['self_play']['ma_bias']

        t = trange(max_exploration_episodes, desc='Self play exploration', leave=True)

        for episode in t:
            t.set_description("Self play exploration")
            t.refresh()
            eval('self.explore_self_play_{}(max_timestep, tolerance, stop_probability)'.format(mode))

        t = trange(max_episodes, desc='Self play training', leave=True)
        train_teacher = True
        last_switch = 0

        if max_timestep_strategy == "auto":
            time_buffer = RingBuffer(capacity=ma_window_length)

            for _ in range(ma_window_length):
                time_buffer.append(ma_default_value)

            max_timestep = int(np.ceil(ma_multiplier * np.mean(time_buffer)))

        for episode in t:
            t.set_description("Self play training")
            t.refresh()

            tA, tB = eval('self.self_play_{}(max_timestep, episode, tolerance, stop_update, alternate, train_teacher)'.format(mode))

            if max_timestep_strategy == "auto":
                time_buffer.append(tA)
                max_timestep = min(int(np.ceil(ma_multiplier * np.mean(time_buffer) + ma_bias)), max_timestep_target)

            if alternate:
                if episode - last_switch >= alternate_step:
                    train_teacher = not(train_teacher)
                    last_switch = episode

            self.apply_noise_decay()

            if episode % test_freq == 0:
                test_mean, test_std = self.test(test_episodes, max_timestep_target, tolerance, render=False)
                self.writer.add_scalars("self_play/{}".format(self.run_id), {'average reward': np.mean(test_mean)}, episode)
                self_play_mean.append(test_mean)
                self_play_std.append(test_std)

                if test_mean >= current_best:
                    current_best = test_mean
                    self.learners.save(self.path + "/models_{}".format(self.run_id))

        return self_play_mean, self_play_std

        max_episodes = self.config['target_play']['episodes']
        max_timestep = self.config['target_play']['max_timestep']
        max_exploration_episodes = self.config['target_play']['exploration_episodes']
        test_freq = self.config['target_play']['test_freq']
        test_episodes = self.config['target_play']['test_episodes']

        t = trange(max_exploration_episodes, desc='Target play exploration', leave=True)
        for episode in t:
            t.set_description("Target play exploration")
            t.refresh()
            self.explore_target_play(max_timestep, tolerance)

        t = trange(max_episodes, desc='Target play training', leave = True)
        for episode in t:
            t.set_description("Target play training")
            t.refresh()
            self.target_play(max_timestep, episode, tolerance)

            if episode % test_freq == 0:
                test_mean, test_std = self.test(test_episodes, max_timestep, tolerance, render=False)
                self.writer.add_scalars("Target_play/{}".format(self.run_id), {'average reward': np.mean(test_mean)}, episode)
                target_play_mean.append(test_mean)
                target_play_std.append(test_std)

        return target_play_mean, target_play_std

    def explore_self_play_repeat(self, tMAX, tolerance, stop_probability=0.5):

        tA = 0
        tB = 0
        solved = False

        seed = random.randint(0, 2 ** 32 - 1)
        np.random.seed(seed)

        s = self.env.reset()
        s_init = copy.deepcopy(s)

        subs_learner = self.get_learners_subpolicies()
        subs_teacher = self.get_teachers_subpolicies()

        teacher_state = {}
        learner_state = {}

        stop_flag = False

        while True:

            tA = tA + 1

            actions_detached = self.teachers.random_act()
            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))

            stop_flag = np.random.rand() > stop_probability

            if stop_flag == True or tA >= tMAX:
                teacher_state['s'] = copy.deepcopy(s)
                teacher_state['s_t'] = copy.deepcopy(s_t)
                teacher_state['a'] = copy.deepcopy(actions_detached)
                teacher_state['d'] = True
                s = s_t
                break

            obs = list(np.hstack((np.array(s_init), np.array(s))))
            obs_t = list(np.hstack((np.array(s_init), np.array(s_t))))

            self.teachers.push_sample(obs, actions_detached, [0] * self.env.n, False, obs_t, subs_teacher)
            s = s_t

        #bases_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
        bases_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

        np.random.seed(seed)

        s = self.env.reset()

        if self.shuffle_self_play:
            np.random.shuffle(bases_pos)

        s = utils.alter_state(s, bases_pos)

        save_s = None
        save_s_t = None

        while True:

            tB = tB + 1
            actions_detached = self.learners.random_act()
            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))
            s_t = utils.alter_state(s_t, bases_pos)

            #agents_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
            agents_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

            if utils.is_solved(bases_pos, agents_pos, tolerance) is True:
                solved = True
            else:
                solved = False

            if tA + tB >= tMAX or solved is True:
                learner_state['s'] = copy.deepcopy(s)
                learner_state['s_t'] = copy.deepcopy(s_t)
                learner_state['a'] = copy.deepcopy(actions_detached)
                learner_state['d'] = solved
                break

            reward = 0

            self.learners.push_sample(s, actions_detached, [0] * self.env.n, solved, s_t, subs_learner)

            s = s_t

        if solved is False:
            tB = tMAX - tA

        R_A = [self.self_play_gamma * max(0, tB - tA)] * self.env.n
        R_B = [self.self_play_gamma * -1 * tB] * self.env.n

        obs = list(np.hstack((np.array(s_init), np.array(teacher_state['s']))))
        obs_t = list(np.hstack((np.array(s_init), np.array(teacher_state['s_t']))))

        self.teachers.push_sample(obs, teacher_state['a'], R_A, teacher_state['d'], obs_t, subs_teacher)
        self.learners.push_sample(learner_state['s'], learner_state['a'], R_B, solved, learner_state['s_t'], subs_learner)

    def explore_target_play(self, max_timestep, tolerance):

        step_count = 0
        Done = False
        timestep = 0

        s = self.env.reset()

        while timestep < max_timestep and Done is False:
            subs = self.get_learners_subpolicies()
            timestep = timestep + 1
            actions = self.learners.random_act()
            s_t, r, done, i = self.env.step(copy.deepcopy(actions))
            Done = True in done

            Done = False

            #bases_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
            #agents_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])

            bases_pos = np.array([copy.deepcopy(base.get_pos()) for base in self.env.landmarks])
            agents_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

            is_solved = utils.is_solved(bases_pos, agents_pos, tolerance)

            if is_solved is True:
                r = [1] * self.env.n
            else:
                r = [0] * self.env.n

            if timestep >= max_timestep or is_solved:
                Done = True

            self.learners.push_sample(s, actions, r, Done, s_t, subs)
            s = s_t

    def self_play_repeat(self, max_timestep, episode, tolerance, stop_update, alternate, train_teacher):
        tA = 0
        tB = 0

        seed = random.randint(0, 2 ** 32 - 1)
        np.random.seed(seed)

        s = self.env.reset()
        s_init = copy.deepcopy(s)

        #start_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
        start_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

        subs_learner = self.get_learners_subpolicies()
        subs_teacher = self.get_teachers_subpolicies()
        teacher_state = {}
        learner_state = {}
        stop_flag = False

        while True:

            tA = tA + 1

            input = torch.Tensor(np.hstack((np.array(s_init), np.array(s))))
            actions_detached = self.teachers.act(input, subs_teacher)

            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))

            stop_input = np.hstack((s_init, s)).flatten()
            stop_action, stop_old_action_prob, stop_value = self.stop.act(torch.Tensor(stop_input))
            #stop_old_action_prob, stop_action = self.stop.act(torch.Tensor(stop_input))

            stop_flag = bool(stop_action.item())

            self.stop.memory.states.append(stop_input)
            #self.stop.memory.old_action_probs.append(stop_old_action_prob)
            self.stop.memory.log_prob.append(stop_old_action_prob)
            self.stop.memory.actions.append(stop_action)
            self.stop.memory.values.append(stop_value)

            if stop_flag == True or tA >= max_timestep:
                teacher_state['s'] = copy.deepcopy(s)
                teacher_state['s_t'] = copy.deepcopy(s_t)
                teacher_state['a'] = copy.deepcopy(actions_detached)
                teacher_state['d'] = True
                s = s_t
                break

            self.stop.memory.rewards.append(0)
            self.stop.memory.dones.append(False)
            #self.stop.memory.is_terminals.append(False)

            obs = list(np.hstack((np.array(s_init), np.array(s))))
            obs_t = list(np.hstack((np.array(s_init), np.array(s_t))))

            self.teachers.push_sample(obs, actions_detached, [0] * self.env.n, False, obs_t, subs_teacher)

            s = s_t

        #bases_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
        bases_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

        distances = []
        for count in range(len(start_pos)):
            distances.append(np.linalg.norm(bases_pos[count] - start_pos[count]))

        self.writer.add_scalars("Teacher distance {}".format(self.run_id), {'Teacher {}'.format(index): distances[index] for index in range(self.env.n)}, episode)

        np.random.seed(seed)

        s = self.env.reset()

        if self.shuffle_self_play:
            np.random.shuffle(bases_pos)

        s = utils.alter_state(s, bases_pos)

        while True:

            tB = tB + 1

            actions_detached = self.learners.act(s, subs_learner)

            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))
            s_t = utils.alter_state(s_t, bases_pos)

            #agents_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
            agents_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

            if utils.is_solved(bases_pos, agents_pos, tolerance):
                solved = True
            else:
                solved = False

            if tA + tB >= max_timestep or solved is True:
                learner_state['s'] = copy.deepcopy(s)
                learner_state['s_t'] = copy.deepcopy(s_t)
                learner_state['a'] = copy.deepcopy(actions_detached)
                learner_state['d'] = solved
                break

            reward = 0

            #changed reward
            self.learners.push_sample(s, actions_detached, [0] * self.env.n, False, s_t, subs_learner)
            s = s_t

        if not solved:
            tB = max_timestep - tA

        distances = []
        for count in range(len(bases_pos)):
            distances.append(np.linalg.norm(bases_pos[count] - agents_pos[count]))

        self.writer.add_scalars("Learner distance {}".format(self.run_id), {'Learner {}'.format(index): distances[index] for index in range(self.env.n)}, episode)

        R_A = [self.self_play_gamma * max(0, tB - tA)] * self.env.n
        R_B = [self.self_play_gamma * -1 * tB] * self.env.n

        obs = list(np.hstack((np.array(s_init), np.array(teacher_state['s']))))
        obs_t = list(np.hstack((np.array(s_init), np.array(teacher_state['s_t']))))

        self.teachers.push_sample(obs, teacher_state['a'], R_A, teacher_state['d'], obs_t, subs_teacher)
        self.learners.push_sample(learner_state['s'], learner_state['a'], R_B, learner_state['d'], learner_state['s_t'], subs_learner)

        self.stop.memory.rewards.append(R_A[0])
        self.stop.memory.dones.append(True)
        #self.stop.memory.is_terminals.append(True)

        self.writer.add_scalars("Self play episode time {}".format(self.run_id), {'ALICE TIME': tA, 'BOB TIME': tB, 'MAX TIME': max_timestep}, episode)
        self.writer.add_scalars("Self play rewards {}".format(self.run_id), {"ALICE REWARD" : R_A[0], 'BOB REWARD': R_B[0]}, episode)

        print("TA : {} TB : {} RA : {} RB {} {}".format(tA, tB, R_A, R_B, "SOLVED" if solved else ""))

        if not alternate or train_teacher is True:
            for _ in range(tA):
                self.teachers.train(subs_teacher)

            if episode % stop_update == 0:
                self.stop.update()

        if not alternate or train_teacher is False:
            for _ in range(tB):
                self.learners.train(subs_learner)

        return tA, tB

    def target_play(self, max_timestep, episode, tolerance):

        timestep = 1

        subs = self.get_learners_subpolicies()
        s = self.env.reset()
        bases_pos = [entity.state.p_pos for entity in self.env.world.landmarks]
        Done = False

        while timestep <= max_timestep and Done is False:

            timestep = timestep + 1

            actions_detached = self.learners.act(s, subs)
            s_t, _, _ , _ = self.env.step(copy.deepcopy(actions_detached))
            #Done = True in done

            #agents_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
            agents_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

            is_solved = utils.is_solved(bases_pos, agents_pos, tolerance)

            if is_solved is True:
                r = [1] * self.env.n
            else:
                r = [0] * self.env.n

            if timestep >= max_timestep or is_solved:
                Done = True

            self.learners.push_sample(s, actions_detached, r, Done, s_t, subs)
            self.learners.train(subs)

            s = s_t

    def test(self, n_episodes, max_episode_timestep, tolerance, render=False):
        results = []

        for episode in range(n_episodes):

            episode_reward = 0

            Done = False
            timestep = 0
            subs = self.get_learners_subpolicies()

            s = self.env.reset()
            #bases_pos = [entity.state.p_pos for entity in self.env.world.landmarks]
            bases_pos = np.array([copy.deepcopy(base.get_pos()) for base in self.env.landmarks])

            while timestep < max_episode_timestep and Done is False:
                timestep = timestep + 1
                actions = []

                actions_detached = self.learners.act(s, subs, noise=False)
                s_t, _, _, _ = self.env.step(copy.deepcopy(actions_detached))

                #Done = True in done

                #agents_pos = np.array([copy.deepcopy(agent.state.p_pos) for agent in self.env.world.agents])
                agents_pos = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])

                is_solved = utils.is_solved(bases_pos, agents_pos)

                if is_solved is True:
                    r = [1] * self.env.n
                else:
                    r = [0] * self.env.n

                if timestep >= max_episode_timestep or is_solved:
                    Done = True

                if render:
                    time.sleep(0.25)
                    self.env.render(mode="human")

                episode_reward += r[0]

                s = s_t

            results.append(episode_reward)

        return np.mean(results, axis=0) , np.std(results, axis=0)

