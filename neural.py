import numpy as np
from neuron import *

class Rewarder:
    pass


class NerveBall:
    def __init__(self, rewarder, inputs=[], outputs=[], size=100, params=NeuronalParams(), positioner=None):
        self.rewarder = rewarder
        self.params = params
        self.hidden = []
        self.neurons = []
        # self.neurons = [Neuron(params)] * size

        temp_neurons = []

        for i in range(size):
            neuron = None
            if positioner is not None:
                neuron = Neuron(params, name='n' + str(i), position=positioner(i))
            else:
                neuron = Neuron(params, name='n' + str(i))
            temp_neurons.append(neuron)

        for pre in temp_neurons:
            for post in temp_neurons + outputs:
                if pre is post: continue

                # This could be tweaked
                if pre.distance(post) < pre.params.synaptic_length:
                    pre.dendrites.append(Dendrite(post))

            if len(pre.dendrites) > 0:
                self.neurons.append(pre)
                self.hidden.append(pre)

        self.outputs = outputs
        self.neurons += outputs
        self.inputs = inputs

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

        # run for 2 cycles
        last_active = 0
        while True:
            for neuron in self.neurons:
                neuron.propagate(t)

            active = len(self.active())
            if active == last_active:
                break
            last_active = active

        # if we are doing poorly and nothing is activated
        # act randomly
        # if reinforcement <= 0:
        #     # for neuron in self.hidden:
        #     #     neuron.activation += np.random.random()
        #
        #     if len(self.active(in_set=self.outputs)) == 0:
        #         np.random.choice(self.outputs).activation = self.params.threshold + 0.1
        #
        #     for sensor in self.active(in_set=self.inputs):
        #         for output in self.active(in_set=self.outputs):
        #             path = self.shortest_path(sensor, output)
        #
        #             last = path[0]
        #             for neuron in path:
        #                 neuron.activation = neuron.params.threshold + 0.1


    def shortest_path(self, frum, to, tag=None):

        if tag is None:
            tag = np.random.random()

        frum.tag = tag

        best_path = []
        for dendrite in frum.dendrites:
            post = dendrite.post_synaptic_neuron

            if post.tag is tag:
                continue

            path = [frum]

            if post is to:
                path += [post]
            else:
                path += self.shortest_path(post, to, tag=tag)

            if len(best_path) == 0 or len(path) < len(best_path):
                best_path = path

        #make sure this is actually a path
        for i in range(0, len(best_path) - 1):
            last = best_path[i]
            found = False
            for dendrite in last.dendrites:
                if dendrite.post_synaptic_neuron is best_path[i+1]:
                    found = True
                    break
            assert found

        return best_path

    def reinforce(self):
        reinforcement = self.rewarder.reinforcement

        for neuron in self.active():
            neuron.reinforce(reinforcement)
            # neuron.activation *= self.params.activation_decay

    def reset(self):
        for neuron in self.active():
            neuron.activation = 0 #0.5 * neuron.params.threshold


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
