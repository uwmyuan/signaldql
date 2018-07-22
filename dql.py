# -*- coding: utf-8 -*-
"""
training and testing the dql
create_dqn to create deep q network
train_one_step_dql to train one-step forward
train_simulated_dql to train on simulator/generator
test_dql to test on simulator/generator
test on sumo has sumo dependencies
@author: Yun
"""

import numpy as np
from scipy.stats.kde import gaussian_kde
import time

import matplotlib.pyplot as plt
import itertools
from collections import deque
import random
# import keras framework
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Reshape, Flatten, Add, Activation, Concatenate, Conv1D
from keras.callbacks import TensorBoard, EarlyStopping, CSVLogger
import sys

sys.path.append('../')
# import local functions
from .generator import create_model
import sumo.runner as runner

# problom-specific constant
# TAIL = number of the vehicle upper bound in seconds
TAIL = 300

NUMBIN = TAIL + 1
NUMVEH = 130
# PHASE = number of phases, dimension of action
PHASE = 4

# LG = number of measurable lane groups, 3rd dimension of state
LG = 6

# NUMCLUSTER = number of clusters of states
NUMCLUSTER = 10

LATENT = 100


# %DQN
# Deep Q-Network (DQN) [Mnih et al., 2015] [Roderick et al., 2017]
# memory D keeps a large history of the most recent experiences (s,a,s',r,T)

def evaluate_reward(state):
    s = 0
    for i in range(LG):
        s += np.dot(np.linspace(0, TAIL, TAIL + 1), state[:, i] / sum(state[:, i]))
    return -s * s


def evaluate_performance(state):
    s = 0
    for i in range(LG):
        s += np.dot(np.linspace(0, TAIL, TAIL + 1), state[:, i] / sum(state[:, i]))
    return s


def evaluate_rewards(states):
    rewards = []
    for i in states:
        rewards.append(evaluate_reward(i))
    return np.array(rewards)


def create_dqn():
    # input
    state = Input(shape=(NUMBIN, LG))
    action = Input(shape=(PHASE,))
    state1 = Input(shape=(NUMBIN, LG))
    reward = Input(shape=(1,))
    ctime = Input(shape=(1,))
    statef = Flatten()(state)
    state1f = Flatten()(state1)
    # concatenate
    out = Concatenate()([statef, action, state1f, reward, ctime])
    out = Dense(256)(out)
    out = Activation('relu')(out)
    out = Reshape((16, 16), input_shape=(256,))(out)
    out = Conv1D(3, 3, padding='same')(out)
    out = Flatten()(out)
    out = Dense(128)(out)
    out = Activation('relu')(out)
    out = Dense(1)(out)
    out = Activation('relu')(out)
    # compile
    model = Model(inputs=[state, action, state1, reward, ctime], outputs=[out])
    # dqn_optim = SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer="adam")
    return model


def predict_dqn(dqn, state, action_set, state1s, rewards, t):
    input_states = []
    ts = []
    for i in range(action_set.shape[0]):
        input_states.append(state)
        ts.append(t)
    return dqn.predict([np.array(input_states), action_set, state1s, rewards, np.array(ts)], verbose=0)


def update_dqn(dqn, state, action, state1, reward, t, q_value):
    dqn.train_on_batch([state, action, state1, reward, t], q_value)


def create_action_set(action):
    actions_max = 100
    # green time -10,-5,0,+5,+10

    GREEN_MIN = 10
    GREEN_MAX = 300
    CYCLE_MIN = 40
    CYCLE_MAX = 600
    action_set = []
    last_action = action * actions_max
    displacement = [-10, -5, 0, 5, 10]
    for combination in itertools.product(displacement, displacement, displacement, displacement):
        i1, i2, i3, i4 = combination
        p1 = last_action[0] + i1
        p2 = last_action[1] + i2
        p3 = last_action[2] + i3
        p4 = last_action[3] + i4
        if GREEN_MAX >= p1 >= GREEN_MIN \
                and GREEN_MAX >= p2 >= GREEN_MIN \
                and GREEN_MAX >= p3 >= GREEN_MIN \
                and GREEN_MAX >= p4 >= GREEN_MIN \
                and CYCLE_MAX >= p1 + p2 + p3 + +p4 >= CYCLE_MIN:
            action_set.append(np.array([p1, p2, p3, p4]) / actions_max)
    return np.array(action_set)


def get_next_states(model, action_set, state, cstart):
    input_states = []
    input_cstart = []
    for i in range(len(action_set)):
        input_states.append(state)
        input_cstart.append(cstart)
    return model.predict([np.array(action_set), np.array(input_states), np.array(input_cstart)])


def get_next_states_sumo(action_set, flow):
    delays = []
    for i in range(len(action_set)):
        signal = {
            'cycle': sum(action_set[i]),
            '65-1-57-3': action_set[i][0],
            '65-5-83-7': action_set[i][1],
            '65-5-64-5': action_set[i][2],
            '65-3-57-3': action_set[i][3]
        }
        sumo_result = np.array(runner.simulation(flow, signal))

        delay = []
        for i in range(LG):
            if np.size(sumo_result[i]) == 0:
                empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7, 1e-6]))
            elif np.size(sumo_result[i]) == 1:
                empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7]))
            else:
                empirical_pdf = gaussian_kde(sumo_result[i])
            pdf = []
            for i in range(0, TAIL + 1, 1):
                pdf.append(empirical_pdf.integrate_box_1d(i, i + 1))
            delay.append(np.array(pdf))
        delay = np.transpose(delay)
        delays.append(delay)
    return np.array(delays)


def find_max_q(action_set, state, t, model, dqn, flow, use_sumo=False):
    # generate potential next states
    if use_sumo:
        state1s = get_next_states_sumo(action_set, flow)
    else:
        state1s = get_next_states(model, action_set, state, t)
    state1s = state1s.clip(0)
    # evaluate rewards
    rewards = evaluate_rewards(state1s)
    # predict Q-value
    predicted_q = predict_dqn(dqn, state, np.array(action_set), np.array(state1s), np.array(rewards), t)
    # a*=argmax_{a} Q
    max_index = np.argmax(predicted_q)
    return action_set[max_index], state1s[max_index]


def find_max_q_value(action_set, state, t, model, dqn, flow, use_sumo=False):
    # generate potential next states
    if use_sumo:
        state1s = get_next_states_sumo(action_set, flow)
    else:
        state1s = get_next_states(model, action_set, state, t)
    state1s = state1s.clip(0)
    rewards = evaluate_rewards(state1s)
    predicted_q = predict_dqn(dqn, state, action_set, np.array(state1s), np.array(rewards), t)
    return max(predicted_q)


def choose_action(action_set, this_state, t, model, dqn, EPSILON):
    action = []
    if np.random.random() < EPSILON:
        action = action_set[np.random.choice(len(action_set))]
        next_states = get_next_states(model, action_set, this_state, t)
        next_state = next_states.clip(0)
    else:
        action, next_state = find_max_q(action_set, this_state, t, model, dqn)
    return action, next_state


# experience replay
# store some experience in D
# randomly resample some cases from D to replay
D = deque(maxlen=2000)
batch_size = 64


def replay(dqn, tdqn, GAMA):
    if len(D) < batch_size:
        return False
    else:
        samples = random.sample(D, batch_size)
        input_states = []
        input_actions = []
        state1s = []
        rewards = []
        ts = []
        for i in range(batch_size):
            input_states.append(samples[i][0])
            input_actions.append(samples[i][1])
            state1s.append(samples[i][2])
            rewards.append(samples[i][3])
            ts.append(samples[i][4])
        input_states = np.array(input_states)
        input_actions = np.array(input_actions)
        state1s = np.array(state1s)
        rewards = np.array(rewards)
        ts = np.array(ts)
        q_values = tdqn.predict([input_states, input_actions, state1s, rewards, ts], verbose=1)
        new_q_values = [x + GAMA * y for x, y in zip(rewards, q_values)]
        for i in range(batch_size):
            update_dqn(dqn, np.array([input_states[i]]), np.array([input_actions[i]]), np.array([state1s[i]]),
                       np.array([rewards[i]]), np.array([ts[i]]), np.array([new_q_values[i]]))
        return True


def update_target_dqn(dqn, tdqn):
    weights = dqn.get_weights()
    target_weights = tdqn.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i]
    tdqn.set_weights(target_weights)


# Generator-based Deep Q-learning
def generate_next_state(model, action, state, cstart):
    return model.predict([np.array([action]), np.array([state]), np.array([cstart])])


def train_simulated_dql(model, dqn, tdqn, GAMEPLAY, TIMESTEP, GAMA=0.8):
    loaded_data = np.load("./RLData.npz")
    actions = loaded_data["actions"]
    states = loaded_data["states"]
    states1 = loaded_data["states1"]
    cstarts = loaded_data["cstarts"]
    flows = loaded_data["flows"]
    actions_max = loaded_data["actions_max"]
    states_max = loaded_data["states_max"]
    cstarts_max = loaded_data["cstarts_max"]

    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    EPSILON_MIN = 0.01
    total_rewards = []
    for m in range(GAMEPLAY):
        # initialize
        print('GAMEPALY: ', m)
        action = actions[94]
        state = states[94]
        # loop
        t = cstarts[94]
        r = 0
        # temporary variables
        last_state = state
        last_action = action
        last_t = cstarts[93]
        total_reward = 0
        while t < TIMESTEP:
            # update state
            generated_states1 = generate_next_state(model, action, state, t)
            state = np.squeeze(generated_states1).clip(0)

            # evaluate reward
            reward = evaluate_reward(state)
            total_reward += reward

            # make decision
            action, state1 = choose_action(create_action_set(action), state, t, model, tdqn, EPSILON)

            # output green time
            print(action * actions_max)

            # memorize
            if len(D) == D.maxlen:
                D.popleft()
            D.append([last_state, last_action, state, reward, last_t])

            # experience replay
            if replay(dqn, tdqn, GAMA):
                r += 1

            # decay
            EPSILON = np.maximum(EPSILON * EPSILON_DECAY, EPSILON_MIN)

            # record last state
            last_state = state
            last_action = action
            last_t = t

            # update time
            t = t + sum(action) + 4 * PHASE

            # update target
            if r % 10 == 0:
                update_target_dqn(dqn, tdqn)
        total_rewards.append(total_reward)

    # save DQN
    dqn.save_weights(time.strftime('%Y-%m-%dT%H%M%S') + 'sDQN.h5')
    return total_rewards


def plot_total_rewards(total_rewards):
    plt.figure(figsize=(12, 7))
    plt.plot(total_rewards)
    plt.title("model train vs validation loss")
    plt.ylabel("total reward")
    plt.xlabel("gameplay")
    plt.savefig(time.strftime('%Y-%m-%d %H%M%S') + ' dqn reward plot', dpi=300)


def plot_history(history):
    plt.figure(figsize=(12, 7))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("model train vs validation loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train', 'validation'])
    plt.savefig(time.strftime('%Y-%m-%d %H%M%S') + ' loss plot', dpi=300)


# Deep Q-learning
# one-step-forward learning
def train_one_step_dql(model, dqn, GAMA, use_sumo=False):
    loaded_data = np.load("./RLData.npz")
    actions = loaded_data["actions"]
    states = loaded_data["states"]
    states1 = loaded_data["states1"]
    cstarts = loaded_data["cstarts"]
    flows = loaded_data["flows"]
    actions_max = loaded_data["actions_max"]
    states_max = loaded_data["states_max"]
    cstarts_max = loaded_data["cstarts_max"]

    # the number of iterative training
    EPOCHS = 100
    # tensorboard logger
    tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=10, write_graph=True, write_grads=False,
                     write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    rewards = np.array(evaluate_rewards(states1))
    for i in range(EPOCHS):
        print("EPOCH:", i)
        q_values = []
        # prediction
        for j in range(len(states)):
            action_set = create_action_set(actions[j])
            if action_set.shape[0]:
                q_values.append(
                    find_max_q_value(action_set, states[j], cstarts[j], model, dqn, flows[j], use_sumo=use_sumo))
            else:
                print(j, 'doesn\'t have any feasible action.')
        print("all q_values generated")
        q_values = np.squeeze(np.array(q_values))
        new_q_values = rewards + GAMA * np.array(q_values)
        # training
        history = dqn.fit([states, actions, states1, rewards, cstarts], new_q_values, epochs=1, batch_size=1,
                          validation_split=0.25, verbose=1, callbacks=[tb])
        # for testing use
        # evals=model.evaluate([dataX,actionIn], actionOut)
        # print('loss:'+str(evals[0])+'\n'+'acc:'+str(evals[1]))
        # #plot history
        # plot_history(history)
        # if i % 10 == 0:
        #    dqn.save(time.strftime('%Y-%m-%dT%H%M%S') + str(i) + 'dDQN.h5')
    dqn.save_weights(time.strftime('%Y-%m-%dT%H%M%S') + 'dDQN.h5')


def test_dql_one_step(model, dqn):
    loaded_data = np.load("./RLData.npz")
    actions = loaded_data["actions"]
    states = loaded_data["states"]
    states1 = loaded_data["states1"]
    cstarts = loaded_data["cstarts"]
    flows = loaded_data["flows"]
    actions_max = loaded_data["actions_max"]
    states_max = loaded_data["states_max"]
    cstarts_max = loaded_data["cstarts_max"]
    # make decision

    rewards_dqn = []
    rewards_scoot = []
    cycle_dqn = []
    cycle_scoot = []
    for i in range(11000, len(states) - 1):
        action, next_state = find_max_q(create_action_set(actions[i]), states[i + 1], cstarts[i + 1], model, dqn)
        print('action chosen by dqn:')
        print(action * actions_max)
        cycle_dqn.append(sum(action) * actions_max)
        print('reward by dqn:')
        rewards_dqn.append(evaluate_reward(next_state))
        print(rewards_dqn[len(rewards_dqn) - 1])
        print('action chosen by scoot:')
        print(actions[i] * actions_max)
        cycle_scoot.append(sum(actions[i]) * actions_max)
        print('reward by scoot:')
        rewards_scoot.append(evaluate_reward(states1[i]))
        print(rewards_scoot[len(rewards_dqn) - 1])
    cycles = np.array([80, 160, 140, 140])
    synchro_actions = np.array([[20, 20, 20, 20], [35, 35, 53, 36], [35, 37, 47, 20], [33, 27, 52, 27]])
    cycle_synchro = []
    rewards_synchro = []
    for i in range(11000, len(states) - 1):
        if cstarts[i] * cstarts_max < 3600 * 6:
            cycle_synchro.append(cycles[0])
            rewards_synchro.append(evaluate_reward(
                np.squeeze(model.predict([synchro_actions[0:1] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
        elif 7 * 3600 < cstarts[i] * cstarts_max < 10 * 3600:
            cycle_synchro.append(cycles[1])
            rewards_synchro.append(evaluate_reward(
                np.squeeze(model.predict([synchro_actions[1:2] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
        elif 18 * 3600 < cstarts[i] * cstarts_max < 20 * 3600:
            cycle_synchro.append(cycles[2])
            rewards_synchro.append(evaluate_reward(
                np.squeeze(model.predict([synchro_actions[2:3] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
        else:
            cycle_synchro.append(cycles[3])
            rewards_synchro.append(evaluate_reward(
                np.squeeze(model.predict([synchro_actions[3:4] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
    plot_reward(rewards_dqn, rewards_scoot, rewards_synchro)
    plot_cycle(cycle_dqn, cycle_scoot, cycle_synchro)


def plot_reward(rewards_dqn, rewards_scoot, rewards_synchro):
    # plot reward
    plt.figure(figsize=(14, 8))
    plt.plot(rewards_dqn, label='DQL')
    plt.plot(rewards_scoot, label='SCOOT')
    plt.plot(rewards_synchro, label='SYNCHRO')
    plt.ylabel("Reward")
    plt.xlabel("Timestamp")
    plt.legend()
    plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + 'testDQN reward', dpi=300)


def plot_cycle(cycle_dqn, cycle_scoot, cycle_synchro):
    # plot cycle
    plt.figure(figsize=(14, 8))
    plt.plot(cycle_dqn, label='DQL')
    plt.plot(cycle_scoot, label='SCOOT')
    plt.plot(cycle_synchro, label='SYNCHRO')
    plt.ylabel("Cycle length(s)")
    plt.ylim(0, 300)
    plt.xlabel("Timestamp")
    plt.legend()
    plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + 'testDQN cycle', dpi=300)


def test_dql(model, dqn):
    loaded_data = np.load("./RLData.npz")
    actions = loaded_data["actions"]
    states = loaded_data["states"]
    states1 = loaded_data["states1"]
    cstarts = loaded_data["cstarts"]
    flows = loaded_data["flows"]
    actions_max = loaded_data["actions_max"]
    states_max = loaded_data["states_max"]
    cstarts_max = loaded_data["cstarts_max"]
    # DQL

    rewards_dqn = []
    cycle_dqn = []
    sumo_rewards_dqn = []

    # SCOOT
    rewards_scoot = []
    generated_rewards_scoot = []
    cycle_scoot = []
    sumo_rewards_scoot = []
    start_index = 0

    # SYNCHRO
    synchro_actions = np.array([[20, 20, 20, 20], [35, 35, 53, 36], [35, 37, 47, 20], [33, 27, 52, 27]])
    generated_rewards_synchro = []
    sumo_rewards_synchro = []

    for i in range(start_index, len(states) - 1):
        # missing data
        if i in [2771, 2772, 8462, 8463, 8464, 8465, 8466]:
            rewards_dqn.append(0)
            rewards_scoot.append(0)
            generated_rewards_scoot.append(0)
            generated_rewards_synchro.append(0)
            sumo_rewards_scoot.append(0)
            sumo_rewards_synchro.append(0)
            sumo_rewards_dqn.append(0)
            continue

        action, next_state = find_max_q(create_action_set(actions[i]), states[i + 1], cstarts[i + 1], model, dqn)
        sumo_action, sumo_next_state = find_max_q(create_action_set(actions[i]), states[i + 1], cstarts[i + 1], model,
                                                  dqn,
                                                  use_sumo=True)
        # reward by dqn
        rewards_dqn.append(evaluate_performance(next_state))
        sumo_rewards_dqn.append(evaluate_performance(sumo_next_state))

        # reward by scoot:
        rewards_scoot.append(evaluate_performance(states1[i]))
        generated_rewards_scoot.append(
            evaluate_performance(
                np.squeeze(model.predict([actions[i + 1:i + 2], states[i:i + 1], cstarts[i + 1:i + 2]]))))
        flow = {
            '65-1-57-3': flows[i][0],
            '65-1-66-1': flows[i][1],
            '65-5-83-7': flows[i][2],
            '65-5-64-5': flows[i][3],
            '65-3-57-3': flows[i][4],
            '65-7-83-7': flows[i][5]
        }
        signal = {
            'cycle': sum(actions[i + 1] * actions_max),
            '65-1-57-3': actions[i + 1][0] * actions_max,
            '65-5-83-7': actions[i + 1][1] * actions_max,
            '65-5-64-5': actions[i + 1][2] * actions_max,
            '65-3-57-3': actions[i + 1][3] * actions_max
        }
        sumo_result = np.array(runner.simulation(flow, signal))

        delays = []
        for i in range(LG):
            if np.size(sumo_result[i]) == 0:
                empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7, 1e-6]))
            elif np.size(sumo_result[i]) == 1:
                empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7]))
            else:
                empirical_pdf = gaussian_kde(sumo_result[i])
            pdf = []
            for i in range(0, TAIL + 1, 1):
                pdf.append(empirical_pdf.integrate_box_1d(i, i + 1))
            delays.append(np.array(pdf))
        delays = np.transpose(delays)
        sumo_rewards_scoot.append(evaluate_performance(delays))

        # SYNCHRO
        if cstarts[i] * cstarts_max < 3600 * 6:
            generated_rewards_synchro.append(evaluate_performance(
                np.squeeze(model.predict([synchro_actions[0:1] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
            signal = {
                'cycle': sum(synchro_actions),
                '65-1-57-3': synchro_actions[0][0],
                '65-5-83-7': synchro_actions[0][1],
                '65-5-64-5': synchro_actions[0][2],
                '65-3-57-3': synchro_actions[0][3]
            }
            sumo_result = np.array(runner.simulation(flow, signal))

            delays = []
            for i in range(LG):
                if np.size(sumo_result[i]) == 0:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7, 1e-6]))
                elif np.size(sumo_result[i]) == 1:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7]))
                else:
                    empirical_pdf = gaussian_kde(sumo_result[i])
                pdf = []
                for i in range(0, TAIL + 1, 1):
                    pdf.append(empirical_pdf.integrate_box_1d(i, i + 1))
                delays.append(np.array(pdf))
            delays = np.transpose(delays)
            sumo_rewards_synchro.append(evaluate_performance(delays))
        elif 7 * 3600 < cstarts[i] * cstarts_max < 10 * 3600:
            generated_rewards_synchro.append(evaluate_performance(
                np.squeeze(model.predict([synchro_actions[1:2] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
            signal = {
                'cycle': sum(synchro_actions[1]),
                '65-1-57-3': synchro_actions[1][0],
                '65-5-83-7': synchro_actions[1][1],
                '65-5-64-5': synchro_actions[1][2],
                '65-3-57-3': synchro_actions[1][3]
            }
            sumo_result = runner.simulation(flow, signal)
            delays = []
            for i in range(LG):
                if np.size(sumo_result[i]) == 0:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7, 1e-6]))
                elif np.size(sumo_result[i]) == 1:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7]))
                else:
                    empirical_pdf = gaussian_kde(sumo_result[i])
                pdf = []
                for i in range(0, TAIL + 1, 1):
                    pdf.append(empirical_pdf.integrate_box_1d(i, i + 1))
                delays.append(np.array(pdf))
            delays = np.transpose(delays)
            sumo_rewards_synchro.append(evaluate_performance(delays))
        elif 18 * 3600 < cstarts[i] * cstarts_max < 20 * 3600:
            generated_rewards_synchro.append(evaluate_performance(
                np.squeeze(model.predict([synchro_actions[2:3] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
            signal = {
                'cycle': sum(synchro_actions[2]),
                '65-1-57-3': synchro_actions[2][0],
                '65-5-83-7': synchro_actions[2][1],
                '65-5-64-5': synchro_actions[2][2],
                '65-3-57-3': synchro_actions[2][3]
            }
            sumo_result = np.array(runner.simulation(flow, signal))

            delays = []
            for i in range(LG):
                if np.size(sumo_result[i]) == 0:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7, 1e-6]))
                elif np.size(sumo_result[i]) == 1:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7]))
                else:
                    empirical_pdf = gaussian_kde(sumo_result[i])
                pdf = []
                for i in range(0, TAIL + 1, 1):
                    pdf.append(empirical_pdf.integrate_box_1d(i, i + 1))
                delays.append(np.array(pdf))
            delays = np.transpose(delays)
            sumo_rewards_synchro.append(evaluate_performance(delays))
        else:
            generated_rewards_synchro.append(evaluate_performance(
                np.squeeze(model.predict([synchro_actions[3:4] / actions_max, states[i:i + 1], cstarts[i + 1:i + 2]]))))
            signal = {
                'cycle': sum(synchro_actions[3]),
                '65-1-57-3': synchro_actions[3][0],
                '65-5-83-7': synchro_actions[3][1],
                '65-5-64-5': synchro_actions[3][2],
                '65-3-57-3': synchro_actions[3][3]
            }
            sumo_result = np.array(runner.simulation(flow, signal))

            delays = []
            for i in range(LG):
                if np.size(sumo_result[i]) == 0:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7, 1e-6]))
                elif np.size(sumo_result[i]) == 1:
                    empirical_pdf = gaussian_kde(np.append(sumo_result[i], [1e-7]))
                else:
                    empirical_pdf = gaussian_kde(sumo_result[i])
                pdf = []
                for i in range(0, TAIL + 1, 1):
                    pdf.append(empirical_pdf.integrate_box_1d(i, i + 1))
                delays.append(np.array(pdf))
            delays = np.transpose(delays)
            sumo_rewards_synchro.append(evaluate_performance(delays))

    start_cstart = [0, 420, 827, 1216, 1693, 2145, 2554, 2958, 3406, 3835,
                    4270, 4711, 5120, 5541, 5957, 6339, 6702, 7056, 7409, 7935,
                    8347, 8765, 9217, 9624, 10123, 10608, 11110, 11598, 12077]
    end_cstart = [419, 826, 1215, 1692, 2144, 2553, 2927, 3407, 3834, 4269,
                  4710, 5119, 5540, 5956, 6338, 6701, 7053, 7498, 7934, 8346,
                  8764, 9216, 9623, 10122, 10607, 11109, 11597, 12076, 12563]

    sum_reward_dqn = []
    sum_reward_scoot = []
    generated_sum_reward_scoot = []
    sumo_sum_reward_scoot = []
    generated_sum_reward_synchro = []
    sumo_sum_reward_synchro = []
    sumo_sum_reward_dqn = []
    for i in range(29):
        sum_reward_dqn.append(sum(rewards_dqn[start_cstart[i] - start_index:end_cstart[i] - start_index][j] for j in
                                  range(end_cstart[i] - start_cstart[i]) if
                                  8 * 3600 < cstarts[j] * cstarts_max < 9 * 3600))
        sum_reward_scoot.append(sum(rewards_scoot[start_cstart[i] - start_index:end_cstart[i] - start_index][j] for j in
                                    range(end_cstart[i] - start_cstart[i]) if
                                    8 * 3600 < cstarts[j] * cstarts_max < 9 * 3600))
        generated_sum_reward_scoot.append(sum(
            generated_rewards_scoot[start_cstart[i] - start_index:end_cstart[i] - start_index][j] for j in
            range(end_cstart[i] - start_cstart[i]) if 8 * 3600 < cstarts[j] * cstarts_max < 9 * 3600))
        generated_sum_reward_synchro.append(sum(
            generated_rewards_synchro[start_cstart[i] - start_index:end_cstart[i] - start_index][j] for j in
            range(end_cstart[i] - start_cstart[i]) if 8 * 3600 < cstarts[j] * cstarts_max < 9 * 3600))
        sumo_sum_reward_scoot.append(sum(
            sumo_rewards_scoot[start_cstart[i] - start_index:end_cstart[i] - start_index][j] for j in
            range(end_cstart[i] - start_cstart[i]) if 8 * 3600 < cstarts[j] * cstarts_max < 9 * 3600))
        sumo_sum_reward_synchro.append(sum(
            sumo_rewards_synchro[start_cstart[i] - start_index:end_cstart[i] - start_index][j] for j in
            range(end_cstart[i] - start_cstart[i]) if 8 * 3600 < cstarts[j] * cstarts_max < 9 * 3600))
        sumo_sum_reward_dqn.append(sum(
            sumo_rewards_dqn[start_cstart[i] - start_index:end_cstart[i] - start_index][j] for j in
            range(end_cstart[i] - start_cstart[i]) if 8 * 3600 < cstarts[j] * cstarts_max < 9 * 3600))
    np.savez("./RLTest",
             sum_reward_dqn=sum_reward_dqn,
             sum_reward_scoot=sum_reward_scoot,
             generated_sum_reward_scoot=generated_sum_reward_scoot,
             sumo_sum_reward_scoot=sumo_sum_reward_scoot,
             sumo_sum_reward_synchro=sumo_sum_reward_synchro,
             generated_sum_reward_synchro=generated_sum_reward_synchro)
    # plot the sums of rewards day-by-day
    plt.figure(figsize=(14, 7))
    x = np.arange('2017-04-01', '2017-04-30', dtype='datetime64[D]')
    plt.plot(x, sum_reward_dqn, "*-", label='DQL on DNN')
    plt.plot(x, sum_reward_scoot, "o-", label='SCOOT on real data')
    plt.plot(x, generated_sum_reward_scoot, "d-", label='SCOOT on DNN')
    plt.plot(x, sumo_sum_reward_scoot, "x-", label='SCOOT on SUMO')
    plt.plot(x, sumo_sum_reward_synchro, "+-", label='SYNCHRO on SUMO')
    plt.plot(x, generated_sum_reward_synchro, "s-", label='SYNCHRO on DNN')
    plt.ylabel("Total mean delay in peak hours (second)")
    plt.xlabel("Date")
    plt.legend()
    # plt.legend(bbox_to_anchor=(1, 1))
    plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + 'testDQN performance', dpi=300)


def main():
    # load inputs
    filename = "your file name"
    dqn = create_dqn()
    model = create_model()
    model.load_weights(filename)

    # train one-step-forward
    train_one_step_dql(model, dqn, GAMA=0.8, use_sumo=False)

    # train on simulation/generator
    tdqn = create_dqn()
    update_target_dqn(dqn, tdqn)
    train_simulated_dql(model, dqn, tdqn, GAMEPLAY=100, TIMESTEP=86400)

    # test
    test_dql(model, dqn)


if __name__ == '__main__':
    main()
