from pandas import Series
from neuron import Neuron
import sigmoid


# version 0.16

class ZekeNet(object):

    def __init__(self, num_in, num_out, *num_hidden):

        self.num_in = num_in
        self.num_out = num_out
        self.num_hidden = num_hidden
        self.num_layers = 2 + len(num_hidden)

        self.net_type = 'default'

        # Neuron array l, n
        self.layers = []
        self.bias = []

        # setup neurons in layers array
        for l in range(self.num_layers):
            if l == 0:
                layer = [Neuron(active=False, value=float(1))
                         for x in range(self.num_in)]
                self.layers.append(layer)
            elif l == self.num_layers - 1:
                layer = [Neuron() for x in range(self.num_out)]
                self.layers.append(layer)
            else:
                layer = [Neuron() for x in range(self.num_hidden[l - 1])]
                self.layers.append(layer)

        # setup restricted neurons in bias array
        for l in range(self.num_layers - 1):
            self.bias.append(Neuron(active=False, value=float(1)))

        # setup connections
        for l in range(self.num_layers - 1, 0, -1):  # start at final index / output layer
            l_k = l
            l_n = l - 1
            k_layer = self.layers[l_k]
            n_layer = self.layers[l_n]
            n_bias_neuron = self.bias[l_n]
            for k_neuron in k_layer:
                for n_neuron in n_layer:
                    k_neuron.add_inbound_connection(n_neuron)

                k_neuron.add_inbound_connection(n_bias_neuron)

    def set_inputs(self, series):
        if series.size != self.num_in:
            print 'ZekeNet Error: Input mismatch'
            return
        for n in range(self.num_in):
            self.layers[0][n].pre_val = series[n]

    def set_sigmoid_hidden(self, new_sigmoid):
        for l in range(1, self.num_layers):
            if l != self.num_layers - 1:
                for neuron in self.layers[l]:
                    neuron.sigmoid = new_sigmoid

    def set_sigmoid_out(self, new_sigmoid):
        for neuron in self.layers[self.num_layers - 1]:
            neuron.sigmoid = new_sigmoid

    def activate(self):
        for neuron in self.layers[self.num_layers - 1]:
            neuron.activate()

    def get_all_connections(self):
        all_connections = []
        for layer in self.layers:
            for neuron in layer:
                for connection in neuron.inbound_connections:
                    all_connections.append(connection)
        return all_connections

    def mutate_weights_rand(self, multiplier=float(1)):
        for connection in self.get_all_connections():
            connection.mutate_weight_rand(multiplier)

    def print_weights(self):
        num = 0
        for connection in self.get_all_connections():
            print '{}:{}'.format(num, connection.weight)
            num += 1
        print "{} total weights".format(num)

    def get_output(self):
        series = Series([neuron.post_val for neuron in self.layers[len(self.layers) - 1]])
        return series


class LinearClassifier(ZekeNet):

    def __init__(self, num_in, num_out, *num_hidden):
        super(LinearClassifier, self).__init__(num_in, num_out, *num_hidden)
        self.set_sigmoid_hidden(sigmoid.leaky_ReLU)
        self.set_sigmoid_out(sigmoid.logistic)
        self.net_type = 'linclass'

class LinearRegressor(ZekeNet):

    def __init__(self, num_in, num_out, *num_hidden):
        super(LinearRegressor, self).__init__(num_in, num_out, *num_hidden)
        self.set_sigmoid_hidden(sigmoid.leaky_ReLU)
        self.set_sigmoid_out(sigmoid.leaky_ReLU)
        self.net_type = 'linreg'
