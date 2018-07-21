import random


class Connection(object):
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron

        self.weight = random.random()  # initially randomize weight

    def change_weight(self, by):
        self.weight += by

    def pull_forward(self):
        return self.from_neuron.activate() * self.weight

    def mutate_weight_rand(self, multiplier=float(1)):
        positive = random.randrange(0, 2)
        change = random.random() * (1 if positive == 0 else -1) * multiplier
        self.change_weight(change)
