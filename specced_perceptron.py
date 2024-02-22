"""
    specced_perceptron.py

    The Agent and the Target models share a need for variable-architecture Perceptron cores,
    so we modularize the behaviour.
"""

import torch


class SpeccedPerceptron(torch.nn.Module):
    """ Basic MLP: PReLU-activated hidden layers, with an optional output layer """

    def __init__(self, layer_spec, last_layer = None):
        """ Initialize from a list of layer widths """
        super().__init__()

        layers = [f(layer_spec[x], layer_spec[x+1]) # note to self: cure addiction to list comprehensions 
            for x in range(len(layer_spec) - 1) 
            for f in (lambda m,n: torch.nn.Linear(m,n), lambda m,n:torch.nn.PReLU())]

        if last_layer:
            self.perceptron = torch.nn.Sequential(*layers[:-1], last_layer) # Trim the last ReLU and add terminal layer
        else:    
            self.perceptron = torch.nn.Sequential(*layers[:-1]) # Trim the last ReLU


    def forward(self, X):
        return self.perceptron(X)


    @classmethod
    def from_text_spec(cls, spec, in_n, out_n, last_layer = None):
        """ Factory for parsing different kinds of textual specification for hidden layers. """

        layers = []
        if 'hidden_layers' in spec:
            # Manual layer specification
            layers = [in_n, *(spec['hidden_layers']), out_n]
        else:
            # Num and counts given
            hidden_count = spec.get('hidden_count', 2)
            hidden_size = spec.get('hidden_size', 100)
            layers = [in_n, *[hidden_size for c in range(hidden_count)], out_n]

        return cls(layers, last_layer)
