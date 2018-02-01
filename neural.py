import numpy as np


class Rewarder:
    @property
    def reinforcement(self):
        return 0


class Dendrite:
    def __init__(self, neuron):
        self.post_synaptic_neuron = neuron
        self.weight = 0
        self.redefine()

    def redefine(self):
        self.weight = np.random.random(1) - 0.5

    def signal(self, decay):
        self.post_synaptic_neuron.activation += self.weight
        self.weight *= decay

    def modify(self, reinforcement):
        if self.post_synaptic_neuron.is_active:
            self.weight *= 1 + reinforcement


class NeuronalParams:
    def __init__(self, activation_decay=0.2, synaptic_decay=0.001, synaptic_reinforcement=0.01, threshold=1, max_synaptic_length=0.2):
        self.activation_decay = 1 - activation_decay
        self.synaptic_decay = 1 - synaptic_decay
        self.synaptic_reinforcement = synaptic_reinforcement
        self.threshold = threshold
        self.synaptic_length = max_synaptic_length


class Neuron:
    def __init__(self, params, name=None, position=np.array([np.random.normal(), np.random.normal()])):
        self.name = name
        self.position = position
        self.dendrites = []
        self.activation = 0
        self.params = params
        self.update_time = 0

    def distance(self, other):
        delta = self.position - other.position
        return np.sqrt((delta ** 2).sum())

    def step(self, t, reinforcement):
        self.update_time = t

        if self.activation > self.params.threshold:
            for dendrite in self.dendrites:
                dendrite.signal(self.params.synaptic_decay)

                if dendrite.post_synaptic_neuron.is_active:
                    dendrite.modify(self.params.synaptic_reinforcement * reinforcement)
                elif abs(dendrite.weight) < 0.1 * self.params.threshold:
                    dendrite.redefine()

        self.activation *= self.params.activation_decay

    def __str__(self):
        return str('%s: %f' % (self.name, self.activation))

    @property
    def is_active(self):
        return self.activation > self.params.threshold


class NerveBall:
    def __init__(self, rewarder, size=100, params=NeuronalParams()):
        self.rewarder = rewarder
        self.neurons = []
        # self.neurons = [Neuron(params)] * size

        for i in range(size):
            self.neurons.append(Neuron(params, name='n' + str(i)))

        for pre in self.neurons:
            for post in self.neurons:
                if pre is post: continue

                # This could be tweaked
                if pre.distance(post) < pre.params.synaptic_length:
                    pre.dendrites.append(Dendrite(post))

    def active(self):
        all_active = []

        for neuron in self.neurons:
            if neuron.is_active: all_active.append(neuron)

        return all_active

    def step(self, t):
        reinforcement = self.rewarder.reinforcement

        for neuron in self.neurons:
            neuron.step(t, reinforcement)

        for neuron in self.active():
            neuron.activation = neuron.params.threshold * 0.5


if __name__ == '__main__':
    class TestRewarder:
        @property
        def reward(self):
            return 1

    params = NeuronalParams(activation_decay=0.05,
                            synaptic_decay=0.05,
                            synaptic_reinforcement=0.5)

    nb = NerveBall(TestRewarder(), size=5, params=params)

    import time

    for t in range(1000):
        nb.step(t)
        print('[t%d]' % t)

        nb.neurons[0].activation = 1.1

        i = 0
        for n in nb.neurons:
            print('n%d -> %s' % (i, str(n.is_active)))
            i += 1

        # time.sleep(0.1)
