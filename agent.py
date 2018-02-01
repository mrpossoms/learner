import numpy as np
import neural
from neural import NeuronalParams


class Agent:
    def __init__(self, rewarder):
        self.position = np.random.random(2)

        params = NeuronalParams(threshold= 0.5,
                                activation_decay=0.05,
                                synaptic_decay=0.0001,
                                synaptic_reinforcement=1,
                                max_synaptic_length=1)

        self.nb = neural.NerveBall(size=8, rewarder=rewarder, params=params)

        self.sensor_nerves = self.nb.neurons[0:4]
        self.actuator_nerves = self.nb.neurons[4:8]

    def step(self, t, measurements):

        for _ in range(1):
            for i in range(4):
                self.sensor_nerves[i].activation = measurements[i]

            self.nb.step(t)

        action = []

        for neuron in self.actuator_nerves:
            action.append(neuron.activation)

        return action

    @property
    def sensors(self):
        arr = []
        sensors = 4
        dt = np.pi * 2 / sensors
        s = 0.01
        for t in range(0, sensors):
            arr.append(np.array([np.cos(dt * t) * s, np.sin(dt * t) * s]) + self.position)

        return arr
