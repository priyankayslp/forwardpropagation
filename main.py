import numpy as np
weights = np.around(np.randon.uniform(size=6),decimal=2)
bias = np.around(np.randon.uniform(size=3),decimal=2)
print(weights)
print(bias)
x_1 = 0.5
x_2 = 0.85
print('x_1 is {} and x_2 is {}'.format(x_1,x_2))
z_11 = x_1*weights[0]+x_2*weights[1]+bias[0]
print('first node of hidden layer z_1 is {}'.format(z_11))
z_12 = x_1*weights[2]+x_2*weights[3]+bias[1]
print('second node of hidden layer z_2 is {}'.format(z_12))
a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))
a_12= 1.0/1.0+np.exp(-z_12)
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))
z_2=a_11*weights[4]+a_12*weights[5]+bias[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))
a_2 = 1.0/1.0+np.exp(-z_2)
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))



# loop through each layer and randomly initialize the weights and biases associated with each node
# notice how we are adding 1 to the number of hidden layers in order to include the output layer
def initialize_network(input, num_hidden_layers,m, num_nodes_output) :
    # global num_nodes
    num_nodes_previous = input
    network = {}
    for layer in range(num_hidden_layers + 1):
        # determine name of layer
        if layer == num_hidden_layers:
            layer_name = 'output'
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)
            num_nodes = m[layer]

        # initialize weights and biases associated with each node in the current layer
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    num_nodes_previous = num_nodes
    return network

network = initialize_network(5,3,[3,2,3],1)
print(network)
def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

from random import seed
import numpy as np

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

print('The inputs to the network are {}'.format(inputs))

weighted_sum = compute_weighted_sum(inputs,network['layer_1']['node_1']['weights'],network['layer_1']['node_1']['bias'])
sum = np.around(weighted_sum[0],decimals=4)
print(sum)
def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))

print(node_activation(sum))


def forward_propagate(network, inputs):
    layer_inputs = list(inputs)  # start with the input layer as the input to the first hidden layer

    for layer in network:

        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]

            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs  # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions



my_network = initialize_network(5, 3, [2, 3, 2], 3)
inputs = np.around(np.random.uniform(size=5), decimals=2)

predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))






