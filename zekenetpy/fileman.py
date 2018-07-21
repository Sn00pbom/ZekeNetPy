import json
import neuralnetwork

def save_to_file(zekenet, location):
    data = {}
    data['info'] = {}
    data['info']['net_type'] = zekenet.net_type
    # data['info']['num_layers'] = zekenet.num_layers
    data['info']['num_in'] = zekenet.num_in
    data['info']['num_hidden'] = zekenet.num_hidden
    data['info']['num_out'] = zekenet.num_out
    data['weights'] = [connection.weight for connection in zekenet.get_all_connections()]

    with open(location, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def load_from_file(location):
    with open(location) as json_file:
        data = json.load(json_file)
        info = data['info']

        net_type = info['net_type']
        # num_layers = info['num_layers']
        num_in = info['num_in']
        num_hidden = info['num_hidden']
        num_out = info['num_out']

        weights = data['weights']

        types = {'default':neuralnetwork.ZekeNet,
                 'linclass':neuralnetwork.LinearClassifier}
        zekenet = types[net_type](num_in, num_out, *num_hidden)
        all_connections = zekenet.get_all_connections()

        for i in range(len(weights)):
            all_connections[i].weight = weights[i]

        return zekenet


