# -*- coding: utf-8 -*-
"""
training and testing the DNN generator
train to create and train the DNN generator
test to test the generator
test_point to test on a single state
test_benchmark to perform a comprehensive test
@author: Yun
"""

import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import time

# import keras framework
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Reshape, Flatten, Add, Activation
from keras.callbacks import TensorBoard, EarlyStopping, CSVLogger
from keras.backend import tensorflow_backend as K
import scipy as sp
from scipy import stats
from sklearn.metrics import mean_squared_error
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


# % error measures
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) * (y_test - y)))


def R2(y_test, y_true):
    return 1 - ((y_test - y_true) * (y_test - y_true)).sum() / (
        (y_true - y_true.mean()) * (y_true - y_true.mean())).sum()


def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1 - rmse(y_test, y_true) / rmse(y_mean, y_true)


def create_model():
    action = Input(shape=(PHASE,))
    state = Input(shape=(TAIL + 1, LG))
    cstart = Input(shape=(1,))
    width = (TAIL + 1) * LG
    out_action = Dense(width, activation='tanh')(action)
    out_cstart = Dense(width, activation='tanh')(cstart)
    out_state = Flatten()(state)
    out_state = Dense(width, activation='tanh')(out_state)
    out = Add()([out_action, out_cstart, out_state])
    out = Activation('softsign')(out)
    for i in range(2):
        out = Dense(width, activation="softsign")(out)
    out = Reshape((TAIL + 1, LG), input_shape=((TAIL + 1) * LG,))(out)
    model = Model(inputs=[action, state, cstart], outputs=[out])
    return model


def train_model(model):
    batch_size = 128
    epochs = 500
    loaded_data = np.load("./RLData.npz")
    actions = loaded_data["actions"]
    states = loaded_data["states"]
    states1 = loaded_data["states1"]
    cstarts = loaded_data["cstarts"]
    flows = loaded_data["flows"]
    actions_max = loaded_data["actions_max"]
    states_max = loaded_data["states_max"]
    cstarts_max = loaded_data["cstarts_max"]

    def kullback_leibler_divergence(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1)
        y_pred = K.clip(y_pred, K.epsilon(), 1)
        return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

    def myloss(y_true, y_pred, e=0.1):
        return (1 - e) * mean_squared_error(y_true, y_pred) + \
               e * kullback_leibler_divergence(y_true, y_pred)

    model.compile(optimizer='adam',
                  loss=myloss)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=2,
                                   verbose=0)
    csvlogger = CSVLogger(time.strftime('%Y-%m-%dT%H%M%S') + "dnn" + ".log")
    model.fit([actions, states, cstarts],
              states1,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.2,
              callbacks=[csvlogger]
              )
    model.save_weights(time.strftime('%Y-%m-%dT%H%M%S') + "DNN.h5")
    return model


def test_point(model, test_point=10735):
    loaded_data = np.load("./RLData.npz")
    actions = loaded_data["actions"]
    states = loaded_data["states"]
    states1 = loaded_data["states1"]
    cstarts = loaded_data["cstarts"]
    flows = loaded_data["flows"]
    actions_max = loaded_data["actions_max"]
    states_max = loaded_data["states_max"]
    cstarts_max = loaded_data["cstarts_max"]

    generated = model.predict(
        [actions[test_point:test_point + 1], states[test_point:test_point + 1], cstarts[test_point:test_point + 1]])
    test_state(states1[test_point], np.squeeze(generated).clip(0))
    plot_generated_comparision(states1[test_point], np.squeeze(generated))


def plot_generated_comparision(generated1, state1):
    # plot delay for each lane group
    for i in range(LG):
        plt.figure(figsize=(12, 7))
        plt.plot(generated1[:, i], 'o', label='generated state(t+1)')
        plt.plot(state1[:, i], '*', label='real state(t+1)')
        plt.title("mean delay")
        plt.legend(bbox_to_anchor=(1, 1))
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + ' test' + str(i) + ' ' + '.png', dpi=300)


def test_state(state1, generated_state1, sumo_result, over=100, verbose=False):
    s_m = []
    g_m = []
    s_o = []
    g_o = []
    sumo_m = []
    sumo_o = []
    for i in range(LG):
        if verbose:
            print("LG #", i, " KL divergence: ", stats.entropy(state1[:, i], generated_state1[:, i]))
            print("LG #", i, " MSE: ", rmse(state1[:, i], generated_state1[:, i]))
            print("LG #", i, " R22: ", R22(generated_state1[:, i], state1[:, i]))
            print("LG #", i, " R2: ", R2(generated_state1[:, i], state1[:, i]))
        gm = np.dot(np.linspace(0, TAIL, TAIL + 1), generated_state1[:, i])
        sm = np.dot(np.linspace(0, TAIL, TAIL + 1), state1[:, i])
        if verbose:
            print("LG #", i, "difference in average delay:", gm - sm)
        s_m.append(sm)
        g_m.append(gm)
        s_o.append(sum(state1[over:, i]))
        g_o.append(sum(generated_state1[over:, i]))
        if np.size(sumo_result[i]) != 0:
            sumo_o.append(np.size(sumo_result[i][np.where(sumo_result[i] >= over)]) / np.size(sumo_result[i]))
            sumo_m.append(np.mean(sumo_result[i]))
        else:
            sumo_m.append(10e-7)
            sumo_o.append(10e-7)
    return np.array(s_m), np.array(g_m), np.array(s_o), np.array(g_o), np.array(sumo_m), np.array(sumo_o)


def plot_mean_pdf(generated_mean, state_mean, sumo_mean):
    from scipy.stats.kde import gaussian_kde
    bins = np.linspace(0, 60, 100)
    # plot delay for each lane group
    for i in range(LG):
        generated_pdf = gaussian_kde(generated_mean[:, i])
        state_pdf = gaussian_kde(state_mean[:, i])
        sumo_pdf = gaussian_kde(sumo_mean[:, i])
        generated_histogram = []
        state_histogram = []
        sumo_histogram = []
        for j in range(100):
            generated_histogram.append(generated_pdf.integrate_box_1d(j * 0.6, (j + 1) * 0.6))
            state_histogram.append(state_pdf.integrate_box_1d(j * 0.6, (j + 1) * 0.6))
            sumo_histogram.append(sumo_pdf.integrate_box_1d(j * 0.6, (j + 1) * 0.6))
        plt.figure(figsize=(12, 7))
        plt.plot(bins, generated_histogram, 'o', label='Generator')
        plt.plot(bins, state_histogram, '*', label='Real Data')
        plt.plot(bins, sumo_histogram, 'd', label='SUMO')
        plt.title("generator performance")
        plt.xlabel("Mean delay")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + '_pdf_mean_LG#' + str(i) + '_' + '.png', dpi=300)


def plot_prob_pdf(generated_prob, state_prob, sumo_prob):
    from scipy.stats.kde import gaussian_kde
    # plot delay for each lane group
    bins = np.linspace(0, 0.1, 100)
    for i in range(LG):
        generated_pdf = gaussian_kde(generated_prob[:, i])
        state_pdf = gaussian_kde(state_prob[:, i])
        sumo_pdf = gaussian_kde(sumo_prob[:, i])
        generated_histogram = []
        state_histogram = []
        sumo_histogram = []
        for j in range(100):
            generated_histogram.append(generated_pdf.integrate_box_1d(j * 0.001, (j + 1) * 0.001))
            state_histogram.append(state_pdf.integrate_box_1d(j * 0.001, (j + 1) * 0.001))
            sumo_histogram.append(sumo_pdf.integrate_box_1d(j * 0.001, (j + 1) * 0.001))
        plt.figure(figsize=(12, 7))
        plt.plot(bins, sumo_histogram, 'd', label='SUMO')
        plt.plot(bins, generated_histogram, 'o', label='Generator')
        plt.plot(bins, state_histogram, '*', label='Real Data')
        plt.title("generator performance")
        plt.xlabel("Probability of mean delay greater than 100 second")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + '_pdf_p100_LG#' + str(i) + '_' + '.png', dpi=300)


def plot_mean_pdf(generated_mean, state_mean, sumo_mean):
    from scipy.stats.kde import gaussian_kde
    bins = np.linspace(0, 60, 100)
    # plot delay for each lane group
    for i in range(LG):
        generated_pdf = gaussian_kde(generated_mean[:, i])
        state_pdf = gaussian_kde(state_mean[:, i])
        sumo_pdf = gaussian_kde(sumo_mean[:, i])
        generated_histogram = []
        state_histogram = []
        sumo_histogram = []
        for j in range(100):
            generated_histogram.append(generated_pdf.integrate_box_1d(j * 0.6, (j + 1) * 0.6))
            state_histogram.append(state_pdf.integrate_box_1d(j * 0.6, (j + 1) * 0.6))
            sumo_histogram.append(sumo_pdf.integrate_box_1d(j * 0.6, (j + 1) * 0.6))
        plt.figure(figsize=(12, 7))
        plt.plot(bins, generated_histogram, 'o', label='Generator')
        plt.plot(bins, state_histogram, '*', label='Real Data')
        plt.plot(bins, sumo_histogram, 'd', label='SUMO')
        plt.title("generator performance")
        plt.xlabel("Mean delay")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + '_pdf_mean_LG#' + str(i) + '_' + '.png', dpi=300)


def plot_prob_pdf(generated_prob, state_prob, sumo_prob):
    # plot delay for each lane group
    bins = np.linspace(0, 0.1, 100)
    for i in range(LG):
        generated_pdf = gaussian_kde(generated_prob[:, i])
        state_pdf = gaussian_kde(state_prob[:, i])
        sumo_pdf = gaussian_kde(sumo_prob[:, i])
        generated_histogram = []
        state_histogram = []
        sumo_histogram = []
        for j in range(100):
            generated_histogram.append(generated_pdf.integrate_box_1d(j * 0.001, (j + 1) * 0.001))
            state_histogram.append(state_pdf.integrate_box_1d(j * 0.001, (j + 1) * 0.001))
            sumo_histogram.append(sumo_pdf.integrate_box_1d(j * 0.001, (j + 1) * 0.001))
        plt.figure(figsize=(12, 7))
        plt.plot(bins, sumo_histogram, 'd', label='SUMO')
        plt.plot(bins, generated_histogram, 'o', label='Generator')
        plt.plot(bins, state_histogram, '*', label='Real Data')
        plt.title("generator performance")
        plt.xlabel("Probability of mean delay greater than 100 second")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + '_pdf_p100_LG#' + str(i) + '_' + '.png', dpi=300)


def test_benchmark(model):
    loaded_data = np.load("./RLData.npz")
    actions = loaded_data["actions"]
    states = loaded_data["states"]
    states1 = loaded_data["states1"]
    cstarts = loaded_data["cstarts"]
    flows = loaded_data["flows"]
    actions_max = loaded_data["actions_max"]
    states_max = loaded_data["states_max"]
    cstarts_max = loaded_data["cstarts_max"]
    state_mean = []

    generated_mean = []
    state_over100 = []
    generated_over100 = []
    sumo_mean = []
    sumo_over100 = []
    for TEST in range(len(actions) - 1):
        generated = model.predict([actions[TEST:TEST + 1], states[TEST:TEST + 1], cstarts[TEST:TEST + 1]])
        flow = {
            '65-1-57-3': flows[TEST][0],
            '65-1-66-1': flows[TEST][1],
            '65-5-83-7': flows[TEST][2],
            '65-5-64-5': flows[TEST][3],
            '65-3-57-3': flows[TEST][4],
            '65-7-83-7': flows[TEST][5]
        }
        signal = {
            'cycle': sum(actions[TEST]),
            '65-1-57-3': actions[TEST][0],
            '65-5-83-7': actions[TEST][1],
            '65-5-64-5': actions[TEST][2],
            '65-3-57-3': actions[TEST][3]
        }
        sumo_result = np.array(runner.simulation(flow, signal))
        s_m, g_m, s_o, g_o, sumo_m, sumo_o = test_state(states1[TEST], np.squeeze(generated).clip(0), sumo_result)
        state_mean.append(s_m)
        generated_mean.append(g_m)
        state_over100.append(s_o)
        generated_over100.append(g_o)
        sumo_mean.append(sumo_m)
        sumo_over100.append(sumo_o)
    plot_delay_timestamp(np.array(generated_mean), np.array(state_mean))
    plot_prob_timestamp(np.array(generated_over100), np.array(state_over100))

    plot_mean_pdf(np.array(generated_mean), np.array(state_mean), np.array(sumo_mean))
    plot_prob_pdf(np.array(generated_over100), np.array(state_over100), np.array(sumo_over100))


def plot_delay_timestamp(generated_mean, state_mean):
    # plot delay for each lane group
    for i in range(LG):
        plt.figure(figsize=(12, 7))
        plt.plot(generated_mean[:, i], 'o', label='generated state(t+1)')
        plt.plot(state_mean[:, i], '*', label='real state(t+1)')
        plt.title("generator performance")
        plt.legend(bbox_to_anchor=(1, 1))
        plt.xlabel("timestamp of state")
        plt.ylabel("mean of delay")
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + ' testDNN' + str(i) + ' ' + '.png', dpi=300)


def plot_prob_timestamp(generated_prob, state_prob):
    # plot delay for each lane group
    for i in range(LG):
        plt.figure(figsize=(12, 7))
        plt.plot(generated_prob[:, i], 'o', label='generated state(t+1)')
        plt.plot(state_prob[:, i], '*', label='real state(t+1)')
        plt.title("generator performance")
        plt.legend(bbox_to_anchor=(1, 1))
        plt.ylabel("probability of delay>100s")
        plt.xlabel("timestamp of state")
        plt.savefig(time.strftime('%Y-%m-%dT%H%M%S') + ' testDNN' + str(i) + ' ' + '.png', dpi=300)


def train():
    model = create_model()
    model = train_model(model)
    return model


def test(model, weight_file_name):
    model.load_weights(weight_file_name)
    test_point(model)
    test_benchmark(model)


if __name__ == '__main__':
    train()
