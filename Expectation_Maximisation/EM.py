import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math
import copy


class EM:
    dim = (100, 1200)

    def __init__(self, dimension, k, points, sigma):
        self.dimension = dimension
        self.k = k
        self.points = points
        self.sigma = sigma
        self.dataSize = points * k

        # Generating data
        self.generators = []
        for i in range(k):
            generator = np.array([np.random.normal(randint(self.dim[0], self.dim[1]), self.sigma, points) for d in range(dimension)])
            self.generators.append(generator.transpose())

        self.data = np.asarray(self.generators).reshape(k * points, dimension)

        # Generating kernel
        self.kernels = []
        for i in range(k):
            mu = [randint(self.dim[0], self.dim[1]) for i in range(dimension)]
            sigma = [self.sigma for i in range(dimension)]
            self.kernels.append({'mu': mu, 'sigma': sigma, 'prior': 1.0/k})
        self.oldKernels = None

    def iteration(self):
        def probability_kernel(x, kernel):
            probability = np.float64(1)
            for d in range(self.dimension):
                probability *= (np.float64(1) / math.sqrt(np.float64(2) * math.pi * kernel['sigma'][d])) * math.exp(-(x[d] - kernel['mu'][d]) ** 2 / (np.float64(2) * kernel['sigma'][d]))
            return probability

        def probability_sum(x, kernels):
            sum = np.float64(0)
            for kernel in kernels:
                sum += probability_kernel(x, kernel) * kernel['prior']
            return sum

        def weight(x, kernel, kernels):
            denominator = probability_sum(x, kernels)
            # Division 0
            if denominator == 0:
                return np.float64(0)
            return (probability_kernel(x, kernel) * kernel['prior']) / denominator

        # Calcul des poids d'appartenance
        weights = [[weight(x, kernel, self.kernels) for x in self.data] for kernel in self.kernels]

        # Calcul des nouveaux kernels
        self.oldKernels = copy.deepcopy(self.kernels)
        for i in range(self.k):
            kernel = self.kernels[i]
            if sum(weights[i]) != 0:
                for d in range(self.dimension):
                    kernel['mu'][d] = (1 / sum(weights[i])) * sum([weights[i][x] * self.data[x][d] for x in range(self.dataSize)])
                    kernel['sigma'][d] = (1 / sum(weights[i])) * sum([weights[i][x] * abs(self.data[x][d] - kernel['mu'][d]) for x in range(self.dataSize)])
                # kernel['prior'] = sum(weights[i]) / self.dataSize

    def draw_class_1d(self):
        plt.style.use('bmh')
        fig, ax = plt.subplots()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        c = 0

        for generator in self.generators:
            x = generator.transpose()[0]
            ax.scatter(x, np.zeros(len(x)), color=colors[c], label = 'Class ' + str(c))
            c += 1

        ax.set_title("Points by class")
        ax.legend(loc='upper right')
        plt.show()

    def draw_class_2d(self):
        plt.style.use('bmh')
        fig, ax = plt.subplots()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        c = 0

        for generator in self.generators:
            x = generator.transpose()[0]
            y = generator.transpose()[1]
            ax.scatter(x, y, color=colors[c], label = 'Class ' + str(c))
            c += 1

        ax.set_title("Points by class")
        ax.legend(loc='upper right')
        plt.show()

    def draw_class(self):
        if self.dimension == 1:
            self.draw_class_1d()
        elif self.dimension == 2:
            self.draw_class_2d()
        else:
            print("Dimension not allowed xplotting")

    def draw_cluster_1d(self, epoch):
        plt.style.use('bmh')
        fig, ax = plt.subplots()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        c = 0
        data = []

        for generator in self.generators:
            x = generator.transpose()[0]
            ax.scatter(x, np.full(len(x), -20), color=colors[c])
            c += 1

        for kernel in self.kernels:
            data.append(np.random.normal(kernel['mu'][0], kernel['sigma'][0], 10000))

        ax.hist(data, 40, alpha=0.8, label=['Cluster ' + str(i) for i in range(self.k)])

        ax.set_title("Epoch " + str(epoch))
        ax.legend(loc='upper right')
        plt.show()

    def draw_cluster_2d(self, epoch):
        plt.style.use('bmh')
        fig, ax = plt.subplots()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        c = 0
        i = 0

        for kernel in self.kernels:
            ax.scatter(kernel['mu'][0], kernel['mu'][1], s=300, label='Cluster ' + str(i))
            i += 1

        for generator in self.generators:
            x = generator.transpose()[0]
            y = generator.transpose()[1]
            ax.scatter(x, y, color=colors[c])
            c += 1

        ax.set_title("Epoch " + str(epoch))
        ax.legend(loc='upper right', prop={'size': 5})
        plt.show()

    def draw_cluster(self, epoch):
        if self.dimension == 1:
            self.draw_cluster_1d(epoch)
        elif self.dimension == 2:
            self.draw_cluster_2d(epoch)

    def has_converged(self):
        if self.oldKernels is None:
            return False

        for i in range(self.k):
            kernel = self.kernels[i]
            oldKernel = self.oldKernels[i]
            if kernel != oldKernel:
                return False
        return True

    def run(self):
        epoch = 0
        self.draw_class()

        while not self.has_converged():
            print("Epoch ", epoch, " : ")
            for i, kernel in enumerate(self.kernels):
                print("Kernel", i, ": mu : ", kernel['mu'], "    sigma : ", kernel['sigma'])
            print()

            self.draw_cluster(epoch)
            self.iteration()
            epoch += 1

        print("The algorithm has converged.")


em = EM(1, 2, 30, 100)
em.run()
