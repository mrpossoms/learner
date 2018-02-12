import numpy as np
import neural
from neural import NeuronalParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

class Agent:
    def __init__(self, rewarder):
        self.position = np.random.random(2)

        params = NeuronalParams(threshold= 0.8,
                                activation_decay=0.01,
                                synaptic_decay=0.1,
                                synaptic_reinforcement=0.9,
                                max_synaptic_length=2)

        self.sensor_nerves = []
        self.actuator_nerves = []

        dt = 2 * np.pi / 4
        pi_4 = np.pi / 4
        for i in range(4):
            # sp = np.array([np.cos(dt * i), np.sin(dt * i)])
            # ap = np.array([np.cos(dt * i + pi_4), np.sin(dt * i + pi_4)])

            self.sensor_nerves += [neural.Neuron(params, name='s' + str(i))]
            self.actuator_nerves += [neural.Neuron(params, name='a' + str(i))]

        self.nb = neural.NerveBall(size=0, inputs=self.sensor_nerves, outputs=self.actuator_nerves, rewarder=rewarder, params=params)

        self.fig, self.ax = plt.subplots()

    def step(self, t, measurements, plot_net=False):
        for i in range(4):
            self.sensor_nerves[i].activation = measurements[i]

        self.nb.step(t)

        action = []



        for neuron in self.actuator_nerves:
            # action.append(neuron.activation)
            action.append(float(neuron.activation > neuron.params.threshold))


        if plot_net:
            self.ax.clear()

            for neuron in self.nb.neurons:
                x, y = neuron.position

                for dendrite in neuron.dendrites:
                    X = [x, dendrite.post_synaptic_neuron.position[0]]
                    Y = [y, dendrite.post_synaptic_neuron.position[1]]

                    color = 'green'

                    if dendrite.weight < 0:
                        color = 'red'

                    a = abs(dendrite.weight)
                    if a > 1:
                        a = 1
                    l = Line2D(X, Y, color=color, alpha=a, solid_capstyle='projecting')
                    self.ax.add_line(l)

                color = 'green'
                if 's' in str(neuron):
                    color = 'blue'
                if 'n' in str(neuron):
                    color = 'orange'

                a = neuron.activation + 0.25
                if a > 1:
                    a = 1
                if a < 0:
                    a = 0.25
                self.ax.scatter(x, y, c=color, s=50, label=str(neuron), alpha=a, edgecolors='black')

            self.ax.legend()
            self.ax.grid(True)

            self.fig.canvas.draw()
            plt.show(block=False)

        return action

    def reinforce(self):
        self.nb.reinforce()
        self.nb.reset()

    @property
    def sensors(self):
        arr = []
        sensors = 4
        dt = np.pi * 2 / sensors
        s = 0.01
        for t in range(0, sensors):
            arr.append(np.array([np.cos(dt * t) * s, np.sin(dt * t) * s]) + self.position)

        return arr
