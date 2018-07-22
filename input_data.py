# -*- coding: utf-8 -*-
"""
input data from the csv file
run main to create data file RLData.npz
@author: Yun
"""
import pandas as pd
import numpy as np
from scipy.stats.kde import gaussian_kde

# constant
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


def preprocessing_normalization(actions, states, states1, cstarts, actions_max, states_max, cstarts_max):
    return actions / actions_max, states / states_max, states1 / states_max, cstarts / cstarts_max


def input_data_cross_value(filename):
    df = pd.read_csv(filename)
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['cstart']] = df[['cstart']].apply(pd.to_datetime)
    actions = []
    for i in range(len(df)):
        actions.append([df.at[i, '65-1-57-3stagelength'],
                        df.at[i, '65-5-83-7stagelength'],
                        df.at[i, '65-5-64-5stagelength'],
                        df.at[i, '65-3-57-3stagelength']])
    states = []
    vehdelays = ['65-1-57-3vehdelay',
                 '65-1-66-1vehdelay',
                 '65-5-83-7vehdelay',
                 '65-5-64-5vehdelay',
                 '65-3-57-3vehdelay',
                 '65-7-83-7vehdelay']

    for j in range(len(df)):
        delays = []
        for k in range(len(vehdelays)):
            delay = np.zeros(TAIL + 1)
            l = [int(j) for j in df[vehdelays[k]][j].split()]
            for i in range(len(l)):
                if (l[i] <= 0):
                    delay[0] += 1
                else:
                    if (l[i] >= TAIL):
                        delay[TAIL] += 1
                    else:
                        delay[l[i]] += 1
            delays.append(delay)
        states.append([np.array(delays).T])
    cstart = [i.second + i.minute * 60 + i.hour * 3600 for i in df['cstart']]
    l = len(actions)
    actions = actions[1:l]
    states_t = states[0:l - 1]
    states_t1 = states[1:l]
    cstart = cstart[0:l - 1]
    return np.array(actions), np.array(states_t), np.array(states_t1), np.array(cstart)


def input_data_histogram(filename):
    df = pd.read_csv(filename)
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['cstart']] = df[['cstart']].apply(pd.to_datetime)
    actions = []
    for i in range(len(df)):
        actions.append([df.at[i, '65-1-57-3stagelength'],
                        df.at[i, '65-5-83-7stagelength'],
                        df.at[i, '65-5-64-5stagelength'],
                        df.at[i, '65-3-57-3stagelength']])
    states = []
    vehdelays = ['65-1-57-3vehdelay',
                 '65-1-66-1vehdelay',
                 '65-5-83-7vehdelay',
                 '65-5-64-5vehdelay',
                 '65-3-57-3vehdelay',
                 '65-7-83-7vehdelay']
    for j in range(len(df)):
        delays = []
        for k in range(len(vehdelays)):
            delay = np.zeros(TAIL + 1)
            l = [int(j) for j in df[vehdelays[k]][j].split()]
            for i in range(len(l)):
                if l[i] <= 0:
                    delay[0] += 1
                else:
                    if l[i] >= TAIL:
                        delay[TAIL] += 1
                    else:
                        delay[l[i]] += 1
            delays.append(delay)
        states.append(np.array(delays).T)
    cstart = [i.second + i.minute * 60 + i.hour * 3600 for i in df['cstart']]
    l = len(actions)
    actions = actions[1:l]
    states_t = states[0:l - 1]
    states_t1 = states[1:l]
    cstart = cstart[0:l - 1]
    return np.array(actions), np.array(states_t), np.array(states_t1), np.array(cstart)


def input_data_delay_list(filename):
    df = pd.read_csv(filename)
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['cstart']] = df[['cstart']].apply(pd.to_datetime)
    actions = []
    for i in range(len(df)):
        actions.append([df.at[i, '65-1-57-3stagelength'],
                        df.at[i, '65-5-83-7stagelength'],
                        df.at[i, '65-5-64-5stagelength'],
                        df.at[i, '65-3-57-3stagelength']])
    states = []
    vehdelays = ['65-1-57-3vehdelay',
                 '65-1-66-1vehdelay',
                 '65-5-83-7vehdelay',
                 '65-5-64-5vehdelay',
                 '65-3-57-3vehdelay',
                 '65-7-83-7vehdelay']
    for j in range(len(df)):
        delays = []
        for k in range(len(vehdelays)):
            delay = np.zeros(NUMBIN)
            row = np.array([int(j) for j in df[vehdelays[k]][j].split()]).clip(0.1)
            p = 0
            for i in range(len(row)):
                delay[p] = row[i]
                p = p + 1
            delays.append(delay)
        delays = np.transpose(delays)
        states.append([delays])
    cstart = [i.second + i.minute * 60 + i.hour * 3600 for i in df['cstart']]
    l = len(actions)
    actions = actions[1:l]
    states_t = states[0:l - 1]
    states_t1 = states[1:l]
    cstart = cstart[0:l - 1]
    return np.array(actions), np.array(states_t), np.array(states_t1), np.array(cstart)


def input_data_delay_arrival(filename):
    df = pd.read_csv(filename)
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['cstart']] = df[['cstart']].apply(pd.to_datetime)
    actions = []
    for i in range(len(df)):
        actions.append([df.at[i, '65-1-57-3stagelength'],
                        df.at[i, '65-5-83-7stagelength'],
                        df.at[i, '65-5-64-5stagelength'],
                        df.at[i, '65-3-57-3stagelength']])
    states = []
    vehdelays = ['65-1-57-3vehdelay',
                 '65-1-66-1vehdelay',
                 '65-5-83-7vehdelay',
                 '65-5-64-5vehdelay',
                 '65-3-57-3vehdelay',
                 '65-7-83-7vehdelay']
    veharrivals = ['65-1-57-3veharrival',
                   '65-1-66-1veharrival',
                   '65-5-83-7veharrival',
                   '65-5-64-5veharrival',
                   '65-3-57-3veharrival',
                   '65-7-83-7veharrival']
    for j in range(len(df)):
        delays = []
        for k in range(len(vehdelays)):
            delay = np.zeros(NUMBIN)
            vehd = np.array([float(x) for x in df[vehdelays[k]][j].split()]).clip(0.1)
            veha = np.array([int(x * 2) for x in df[veharrivals[k]][j].split()])
            for x, y in zip(veha, vehd):
                if x > TAIL or x <= 0:
                    print("departure time exceeded the interval")
                    continue
                delay[x] = y
            delays.append(delay)
        delays = np.transpose(delays)
        states.append([delays])
    cstart = [i.second + i.minute * 60 + i.hour * 3600 for i in df['cstart']]
    l = len(actions)
    actions = actions[1:l]
    states_t = states[0:l - 1]
    states_t1 = states[1:l]
    cstart = cstart[0:l - 1]
    return np.array(actions), np.array(states_t), np.array(states_t1), np.array(cstart)


def input_data_delay_kde(filename):
    df = pd.read_csv(filename)
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['cstart']] = df[['cstart']].apply(pd.to_datetime)
    actions = []
    for i in range(len(df)):
        actions.append([df.at[i, '65-1-57-3stagelength'],
                        df.at[i, '65-5-83-7stagelength'],
                        df.at[i, '65-5-64-5stagelength'],
                        df.at[i, '65-3-57-3stagelength']])
    states = []
    flows = []
    vehdelays = ['65-1-57-3vehdelay',
                 '65-1-66-1vehdelay',
                 '65-5-83-7vehdelay',
                 '65-5-64-5vehdelay',
                 '65-3-57-3vehdelay',
                 '65-7-83-7vehdelay']
    for j in range(len(df)):
        delays = []
        flow = []
        for k in range(len(vehdelays)):
            l = [float(j) for j in df[vehdelays[k]][j].split()]
            # prevent singular matrix
            l.append(1e-7)
            empirical_pdf = gaussian_kde(l)
            pdf = []
            for i in range(0, TAIL + 1, 1):
                pdf.append(empirical_pdf.integrate_box_1d(i, i + 1))
            delays.append(np.array(pdf))
            flow.append(len(l))
        flows.append(flow)
        delays = np.transpose(delays)
        states.append(delays)
    cstart = [i.second + i.minute * 60 + i.hour * 3600 for i in df['cstart']]
    l = len(actions)
    actions = actions[1:l]
    states_t = states[0:l - 1]
    states_t1 = states[1:l]
    cstart = cstart[0:l - 1]
    return np.array(actions), np.array(states_t), np.array(states_t1), np.array(cstart), np.array(flows)


def input_data_delay_sequence_cnn(filename):
    df = pd.read_csv(filename)
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['cstart']] = df[['cstart']].apply(pd.to_datetime)
    actions = []
    for i in range(len(df)):
        actions.append([df.at[i, '65-1-57-3stagelength'],
                        df.at[i, '65-5-83-7stagelength'],
                        df.at[i, '65-5-64-5stagelength'],
                        df.at[i, '65-3-57-3stagelength']])
    states = []
    vehdelays = ['65-1-57-3vehdelay',
                 '65-1-66-1vehdelay',
                 '65-5-83-7vehdelay',
                 '65-5-64-5vehdelay',
                 '65-3-57-3vehdelay',
                 '65-7-83-7vehdelay']
    for j in range(len(df)):
        delays = []
        for k in range(len(vehdelays)):
            delay_lg = []
            l = [int(j) for j in df[vehdelays[k]][j].split()]
            for i in range(len(l)):
                delay_veh = np.zeros(NUMBIN)
                if (l[i] <= 0):
                    delay_veh[0] = 1
                else:
                    if (l[i] >= TAIL):
                        delay_veh[TAIL] = 1
                    else:
                        delay_veh[l[i]] = 1
                delay_lg.append(delay_lg)
            delays.append(delay_lg)
        states.append(delays)
    cstart = [i.second + i.minute * 60 + i.hour * 3600 for i in df['cstart']]
    l = len(actions)
    actions = actions[1:l]
    states_t = states[0:l - 1]
    states_t1 = states[1:l]
    cstart = cstart[0:l - 1]
    return np.array(actions), np.array(states_t), np.array(states_t1), np.array(cstart)


def input_data_delay_sequence_rnn(filename):
    df = pd.read_csv(filename)
    df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df[['cstart']] = df[['cstart']].apply(pd.to_datetime)
    actions = []
    for i in range(len(df)):
        actions.append([df.at[i, '65-1-57-3stagelength'],
                        df.at[i, '65-5-83-7stagelength'],
                        df.at[i, '65-5-64-5stagelength'],
                        df.at[i, '65-3-57-3stagelength']])
    input_states_t1 = []
    target_states_t1 = []
    states_t = []
    vehdelays = ['65-1-57-3vehdelay',
                 '65-1-66-1vehdelay',
                 '65-5-83-7vehdelay',
                 '65-5-64-5vehdelay',
                 '65-3-57-3vehdelay',
                 '65-7-83-7vehdelay']
    for k in range(len(vehdelays)):
        delays_df_t1 = []
        delays_df_t1_target = []
        delays_df_t = []
        for j in range(len(df)):
            delay_veh_start = []
            delay_veh_end = []
            delay_veh_t = []
            l = [int(q) for q in df[vehdelays[k]][j].split()]
            if j > 0:
                # add [start]
                delay_veh_start.append(TAIL + 3)
            for i in range(len(l)):
                # 0-TAIL -> 0-TAIL
                # <0 -> TAIL+1
                # >TAIL ->TAIL+2
                # start -> TAIL+3
                # end -> TAIL+4
                if l[i] <= 0:
                    if j > 0:
                        delay_veh_start.append(TAIL + 1)
                        delay_veh_end.append(np.array([TAIL + 1]))
                    if j < len(df) - 1:
                        delay_veh_t.append(TAIL + 1)
                else:
                    if l[i] > TAIL:
                        if j > 0:
                            delay_veh_start.append(TAIL + 2)
                            delay_veh_end.append(np.array([TAIL + 2]))
                        if j < len(df) - 1:
                            delay_veh_t.append(TAIL + 2)
                    else:
                        if j > 0:
                            delay_veh_start.append(l[i])
                            delay_veh_end.append(np.array([l[i]]))
                        if j < len(df) - 1:
                            delay_veh_t.append(l[i])
            if j > 0:
                # add [end]
                delay_veh_end.append(np.array([TAIL + 4]))
                while len(delay_veh_end) < NUMVEH:
                    delay_veh_end.append(np.array([0]))
                while len(delay_veh_start) < NUMVEH:
                    delay_veh_start.append(0)
                delays_df_t1.append(np.array(delay_veh_start))
                delays_df_t1_target.append(np.array(delay_veh_end))
            if j < len(df) - 1:
                while len(delay_veh_t) < NUMVEH:
                    delay_veh_t.append(0)
                delays_df_t.append(np.array(delay_veh_t))
        input_states_t1.append(np.array(delays_df_t1))
        target_states_t1.append(np.array(delays_df_t1_target))
        states_t.append(np.array(delays_df_t))
    l = len(actions)
    actions = actions[1:l]
    return np.array(actions), states_t, input_states_t1, target_states_t1


def main():
    actions, states, states1, cstarts, flows = input_data_delay_kde('cross65ganm.csv')
    actions_max = 100
    states_max = np.max([np.max(states), np.max(states1)])
    cstarts_max = np.max(cstarts)
    actions, states, states1, cstarts = preprocessing_normalization(actions, states, states1, cstarts, actions_max,
                                                                    states_max, cstarts_max)
    np.savez("./RLData",
             actions=actions,
             states=states,
             states1=states1,
             cstarts=cstarts,
             flows=flows,
             actions_max=actions_max,
             states_max=states_max,
             cstarts_max=cstarts_max)


if __name__ == '__main__':
    main()
