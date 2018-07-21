from connection import Connection
import sigmoid

class Neuron(object):

    def __init__(self, sigmoid= sigmoid.pass_through, active = True, value = float(0)):
        self.pre_val = value
        self.post_val = value
        self.inbound_connections = []
        self.active = active
        self.sigmoid = sigmoid

    def activate(self):
        if self.active == False: return self.pre_val

        value = 0
        for connection in self.inbound_connections:
            value += connection.pull_forward()
        self.pre_val = value
        self.post_val = self.sigmoid(self.pre_val)

        return self.post_val

    def add_inbound_connection(self, from_neuron):
        connection = Connection(from_neuron, self)
        self.inbound_connections.append(connection)
