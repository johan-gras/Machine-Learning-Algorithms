from numpy import *
from random import randint
import matplotlib.pyplot as plt

dataset =[([[0,0,1,0,0],
            [0,1,0,1,0],
            [1,1,1,1,1],
            [1,0,0,0,1]], 1.0), #A
          ([[0,0,1,1,1],
            [1,1,0,0,0],
            [1,1,0,0,0],
            [0,0,1,1,1]], 0.0)] #C
L_RATE = 0.05


def retine(img):
    row = ravel([img[0]]).tolist()
    row.append(img[1])
    return row


def init_weights(size):
    return [random.uniform(-1.0, 1.0) for i in range(size)]


def propagation(row, weights):
    activation = weights[-1]
    for i in range(len(row)-1):
        activation += weights[i] * row[i]

    #Heaviside activation
    return 1.0 if activation >= 0.0 else 0.0


def train_weights(row, weights, l_rate):
    prediction = propagation(row, weights)
    error = row[-1] - prediction
    weights[-1] = weights[-1] + l_rate * error

    for i in range(len(row)-1):
        weights[i] = weights[i] + l_rate * error * row[i]

    return weights, error**2


def total_error(rows, weights):
    error = 0.0
    for row in rows:
        error += (row[-1] - propagation(row, weights))**2

    return error / len(rows)


def draw_graph(x_array, y_array, title, xlabel, ylabel):
    plt.cla()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.scatter(x_array, y_array, s=100, c='red', marker='^')
    plt.show()


def learning_seq(dataset, l_rate):
    row1 = retine(dataset[0])
    row2 = retine(dataset[1])
    weights = init_weights(len(row1))
    error = 1.0
    epochA = 1
    epochC = 1
    error_array = []

    print("Apprentissage sequenciel de A puis de B : \n")
    print("Poids initiaux : ", weights, '\n')

    while error == 1.0:
        t_error = total_error([row1, row2], weights)
        error_array.append(t_error)
        (weights, error) = train_weights(row1, weights, l_rate)
        print("Epoque ", epochA, ".A : ", error, " (erreur du motif A) / ", t_error, " (erreur total moyenne)\n")
        epochA += 1

    error = 1.0
    while error == 1.0:
        t_error = total_error([row1, row2], weights)
        error_array.append(t_error)
        (weights, error) = train_weights(row2, weights, l_rate)
        print("Epoque ", epochC, ".C : ", error, " (erreur du motif C) / ", t_error, " (erreur total moyenne)\n")
        epochC += 1

    t_error = total_error([row1, row2], weights)
    error_array.append(t_error)
    print("Erreur total moyenne après apprentissage : ", t_error, "\n")
    print("Poids finaux : ", weights, '\n')
    draw_graph(range(1, epochA + epochC), error_array,
               'Performance du réseau à chaque époche de l\'apprentissage séquenciel', 'Epoque', 'Erreur moyenne')

    return weights


def learning_alt(dataset, l_rate):
    row1 = retine(dataset[0])
    row2 = retine(dataset[1])
    weights = init_weights(len(row1))
    error1 = 1.0
    error2 = 1.0
    epoch = 1
    error_array = []

    print("Apprentissage alterné de A et de B : \n")
    print("Poids initiaux : ", weights, '\n')

    while error1 == 1.0 or error2 == 1.0:
        print("Epoque ", epoch, " : ")
        errorT = total_error([row1, row2], weights)
        error_array.append(errorT)
        print("Erreur total moyenne : ", errorT)
        (weights, error1) = train_weights(row1, weights, l_rate)
        print ("Erreur de A : ", error1)
        (weights, error2) = train_weights(row2, weights, l_rate)
        print ("Erreur de C : ", error2, "\n")
        epoch += 1

    print("Poids finaux : ", weights, '\n')
    draw_graph(range(1, epoch), error_array, 'Performance du réseau à chaque époche de l\'apprentissage alterné',
               'Epoque', 'Erreur moyenne')

    return weights


def noise_row(row, noise):
    for i in range(len(row)-1):
        if randint(0, 100) < noise/2:
            row[i] = int(not row[i])
    return row


def noise_generalisation(dataset, weights):
    print("Tests de capacité de généralisations pour des entrées bruitées en cours...")
    percent_array = range(0, 101)
    error_array = []
    for noise in percent_array:
        batch_error = 0.0
        for i in range(1000):
            row1 = noise_row(retine(dataset[0]), noise)
            row2 = noise_row(retine(dataset[1]), noise)
            batch_error += total_error([row1, row2], weights)
        avg_error = batch_error / 1000.0
        error_array.append(avg_error)
    draw_graph(percent_array, error_array, 'Performance du réseau en fonction du bruit', 'Pourcentage de bruit',
               'Erreur moyenne')


learning_seq(dataset, L_RATE)
weights = learning_alt(dataset, L_RATE)

noise_generalisation(dataset, weights)

