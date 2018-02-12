import numpy as np


class Rewarder:
    pass

class Dendrite:
    def __init__(self, neuron):
        self.post_synaptic_neuron = neuron
        self.weight = np.random.random(1)# - 0.5
        self.redefine()

    def redefine(self):
        s = self.weight / abs(self.weight)
        self.weight = np.random.random(1) #* s
        # self.weight = np.random.random(1) - 0.5

    def signal(self):
        self.post_synaptic_neuron.activation += self.weight
        # self.weight *= decay

    def modify(self, reinforcement):
        if self.post_synaptic_neuron.is_active:
            self.weight *= reinforcement

            if self.weight > 1:
                self.weight = 1

class NeuronalParams:
    def __init__(self, activation_decay=0.2, synaptic_decay=0.001, synaptic_reinforcement=0.01, threshold=1, max_synaptic_length=0.2):
        self.activation_decay = 1 - activation_decay
        self.synaptic_decay = 1 - synaptic_decay
        self.synaptic_reinforcement = synaptic_reinforcement
        self.threshold = threshold
        self.synaptic_length = max_synaptic_length


class Neuron:
    def __init__(self, params, name=None, position=None):
        self.name = name

        if position is None:
            self.position = np.random.randn(2)
        else:
            self.position = position

        self.dendrites = []
        self.activation = 0
        self.params = params
        self.update_time = 0

    def distance(self, other):
        delta = self.position - other.position
        return np.sqrt((delta ** 2).sum())

    def reinforce(self, reinforcement):
        for dendrite in self.dendrites:
            if self.activation > self.params.threshold:
                if dendrite.post_synaptic_neuron.is_active:
                    dendrite.modify(self.params.synaptic_reinforcement + reinforcement)
            # else:
            # dendrite.weight *= self.params.synaptic_decay

            # if abs(dendrite.weight) < 0.01:
            #     dendrite.redefine()

    def propagate(self, t):
        self.update_time = t

        for dendrite in self.dendrites:
            if self.activation > self.params.threshold:
                dendrite.signal()

            # if abs(dendrite.weight) < 0.01:
            #     dendrite.redefine()

    def __str__(self):
        return str('%s: %f' % (self.name, self.activation))

    @property
    def is_active(self):
        return self.activation > self.params.threshold


class NerveBall:
    def __init__(self, rewarder, inputs=[], outputs=[], size=100, params=NeuronalParams()):
        self.rewarder = rewarder
        self.params = params
        self.hidden = []
        self.neurons = []
        # self.neurons = [Neuron(params)] * size

        for i in range(size):
            neuron = Neuron(params, name='n' + str(i))
            self.hidden.append(neuron)
            self.neurons.append(neuron)

        for pre in self.neurons:
            for post in self.neurons + outputs:
                if pre is post: continue

                # This could be tweaked
                if pre.distance(post) < pre.params.synaptic_length:
                    pre.dendrites.append(Dendrite(post))

        self.outputs = outputs
        self.neurons += outputs

        for pre in inputs:
            for post in self.neurons:
                if pre.distance(post) < pre.params.synaptic_length:
                    pre.dendrites.append(Dendrite(post))

        self.neurons = inputs + self.neurons

    def inhibited(self):
        set = []

        for neuron in self.neurons:
            if not neuron.is_active: set.append(neuron)

        return set

    def active(self, in_set=None):
        set = []

        if in_set is None:
            in_set = self.neurons

        for neuron in in_set:
            if neuron.is_active: set.append(neuron)

        return set

    def step(self, t):
        reinforcement = self.rewarder.reinforcement

        for neuron in self.neurons:
            neuron.propagate(t)

        # if we are doing poorly and nothing is activated
        # act randomly
        if reinforcement <= 0:
            for neuron in self.hidden:
                neuron.activation += np.random.random() * 0.2

            if len(self.active(in_set=self.outputs)) == 0:
                np.random.choice(self.outputs).activation = self.params.threshold + 0.1

    def reinforce(self):
        reinforcement = self.rewarder.reinforcement

        for neuron in self.active():
            neuron.reinforce(reinforcement)
            neuron.activation *= self.params.activation_decay

    def reset(self):
        for neuron in self.active():
            neuron.activation = 0.5 * neuron.params.threshold

        for neuron in self.inhibited():
            neuron.activation = 0


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
