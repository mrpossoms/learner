import numpy as np


class NeuronalParams:
    def __init__(self, activation_decay=0.2, synaptic_decay=0.001, synaptic_reinforcement=0.01, threshold=1, max_synaptic_length=0.2):
        self.activation_decay = 1 - activation_decay
        self.synaptic_decay = 1 - synaptic_decay
        self.synaptic_reinforcement = synaptic_reinforcement
        self.threshold = threshold
        self.threshold_decay = 1
        self.synaptic_length = max_synaptic_length


class Dendrite:
    def __init__(self, neuron):
        self.post_synaptic_neuron = neuron

        self.weight = np.random.random(1) * 0.1 + 0.01# - 0.5
        # self.redefine()

    def redefine(self):
        self.weight = np.random.random(1)
        # self.weight = np.random.random(1) - 0.5

    def signal(self):
        assert self.weight> 0
        self.post_synaptic_neuron.activation += self.weight
        # self.weight *= decay

    def modify(self, reinforcement):
        if self.post_synaptic_neuron.is_active:
            self.weight += 0.01 * reinforcement

            # if self.weight > 1:
            #     self.weight = 1 - (np.random.random() * 10E-5)

            if self.weight <= 10E-5:
                self.weight = 10E-5 + np.random.random(1) * 10E-5


class Neuron:
    def __init__(self, params, name=None, position=None):
        self.name = name

        if position is None:
            self.position = np.random.randn(2)
        else:
            self.position = position

        self.dendrites = []
        self.activation = 0
        self.threshold = params.threshold
        self.params = params
        self.update_time = 0
        self.fired = 0
        self.tag = 0

    def distance(self, other):
        delta = self.position - other.position
        return np.sqrt((delta ** 2).sum())

    def reinforce(self, reinforcement):
        for dendrite in self.dendrites:
            if self.activation > self.params.threshold:
                if dendrite.post_synaptic_neuron.is_active:
                    dendrite.modify(reinforcement) #self.params.synaptic_reinforcement + reinforcement)
            # else:
            # dendrite.weight *= self.params.synaptic_decay

            # if abs(dendrite.weight) < 0.01:
            #     dendrite.redefine()

    def propagate(self, t):
        self.update_time = t
        # decay = self.params.threshold_decay

        # self.threshold *= decay
        if self.activation > self.threshold:
            self.fired = t
            self.activation = 0

            for dendrite in self.dendrites:
                dendrite.signal()
                # self.threshold = (1 - decay) * self.params.threshold

            # if abs(dendrite.weight) < 0.01:
            #     dendrite.redefine()

    def __str__(self):
        return str('%s: %f' % (self.name, self.activation))

    @property
    def is_active(self):
        return self.update_time == self.fired
