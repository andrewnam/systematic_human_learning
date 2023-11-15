import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 output_size,
                 activation=F.relu,
                 bias=True,
                 skip_connections=False
                 ):
        super().__init__()

        assert type(input_size) is int or type(input_size) is list
        assert hidden_layer_size is None or type(hidden_layer_size) is int or type(hidden_layer_size) is list
        assert type(output_size) is int or type(output_size) is list

        if type(input_size) is int:
            input_size = [input_size]
        if hidden_layer_size is None:
            hidden_layer_size = []
        elif type(hidden_layer_size) is int:
            hidden_layer_size = [hidden_layer_size]
        if type(output_size) is int:
            output_size = [output_size]

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.activation = activation
        self.skip_connections = skip_connections

        self.layers = nn.ModuleList()
        prev_layer_size = sum(input_size)
        for size in hidden_layer_size:
            self.layers.append(nn.Linear(prev_layer_size, size, bias=bias))
            prev_layer_size = size
        self.layers.append(nn.Linear(prev_layer_size, sum(output_size), bias=bias))

    def forward(self, *inputs):
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = torch.cat(inputs, dim=-1)

        for layer in self.layers:
            x_in = x
            x = layer(x)
            if self.skip_connections:
                x = x + x_in
            if layer is not self.layers[-1]:
                x = self.activation(x)

        if len(self.output_size) == 1:
            return x

        outputs = []
        index = 0
        for size in self.output_size:
            outputs.append(x[..., index:index + size])
            index += size
        return outputs
