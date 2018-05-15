import random
import math
from collections import namedtuple
import matplotlib.pyplot as plt

L_RATE = 0.09
Neuron = namedtuple('Neuron', ['synapse', 'potential', 'output'])

neuronset = None
dataset = None
sorted_data = [False for i in range(20)]
epoch = 0


def InitialiseNeurons():
    global neuronset
    neuronset = [Neuron(synapse = (math.cos((i*2*math.pi)/20)*100 + 100, math.sin((i*2*math.pi)/20)*100 + 100), potential = 0, output = 0) for i in range(20)]


def InitialiseSet():
    global dataset
    dataset = [(random.randint(0, 200), random.randint(0, 200)) for i in range(20)]


def SortData():
    not_sorted = []
    for i in range(20):
        if not sorted_data[i]:
            not_sorted.append((dataset[i], i))
    (data, index) = not_sorted[random.randint(0, len(not_sorted)-1)]
    sorted_data[index] = True
    return data


def CalculatePotential(vector):
    for i in range(20):
        neuronset[i] = neuronset[i]._replace(potential = math.sqrt((vector[0] - neuronset[i].synapse[0])**2 + (vector[1] - neuronset[i].synapse[1])**2))


def CalculateOutputs():
    max_out = (0, -1)

    for i in range(20):
        out = 1/(1 + neuronset[i].potential)
        neuronset[i] = neuronset[i]._replace(output=out)
        if out > max_out[0]:
            max_out = (out, i)

    return max_out[1]


def Neighbour_Coef(winner, index):
    diff = math.sqrt((winner - index)**2)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.6
    if diff == 2:
        return -0.1
    return 0.0


def UpdateSynapse(winner, vector):
    for i in range(20):
        neighbour_coef = Neighbour_Coef(winner, i)
        if neighbour_coef != 0.0:
            neuronset[i] = neuronset[i]._replace(synapse= (neuronset[i].synapse[0] + L_RATE * neighbour_coef * (vector[0] - neuronset[i].synapse[0]),
                                                           neuronset[i].synapse[1] + L_RATE * neighbour_coef * (vector[1] - neuronset[i].synapse[1])))

def Draw():
    plt.cla()
    plt.title('Kohonen (epochs ' + str(epoch) + ')')
    plt.xlabel('x')
    plt.ylabel('y')

    i = []
    j = []
    for data in dataset:
        (x, y) = data
        i.append(x)
        j.append(y)
    plt.scatter(i, j, s=100, c='red', marker='^')

    i = []
    j = []
    for neuron in neuronset:
        (x, y) = neuron.synapse
        i.append(x)
        j.append(y)

    for n in range(19):
        plt.plot(i[n:n+2], j[n:n+2], 'g')
    plt.scatter(i, j, s=60, c='b')
    
    plt.pause(0.001)


InitialiseNeurons()
InitialiseSet()

plt.ion()
Draw()

while True:
    epoch += 1
    sorted_data = [False for i in range(20)]

    for i in range(20):
        vector_chosen = SortData()
        CalculatePotential(vector_chosen)
        max_indice = CalculateOutputs()
        UpdateSynapse(max_indice, vector_chosen)

        Draw()
