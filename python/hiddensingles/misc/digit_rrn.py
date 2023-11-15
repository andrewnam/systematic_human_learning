import torch
import torch.nn as nn
import numpy as np
import itertools

from . import torch_utils as tu
from . import utils
from .nnmodule import nnModule


class DigitRRN(nnModule):

    def __init__(self,
                 dim_x=3,
                 dim_y=3,
                 hidden_vector_size=64,
                 message_size=64):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.hidden_vector_size = hidden_vector_size
        self.message_size = message_size

        self.message_linear = nn.Linear(hidden_vector_size + hidden_vector_size, message_size)
        self.update_node_linear = nn.Linear(1 + message_size, hidden_vector_size)
        self.update_node_lstm = nn.LSTM(hidden_vector_size, hidden_vector_size)
        self.output_linear = nn.Linear(hidden_vector_size, 1)

        # reusable tensors that can be pre-computed
        neighbors = get_neighbors(dim_x, dim_y)
        self.neighbors = torch.where(neighbors)[1].view(self.num_nodes, -1)

    @property
    def max_digit(self):
        return self.dim_x * self.dim_y

    @property
    def num_cells(self):
        return self.max_digit ** 2

    @property
    def num_nodes(self):
        return self.max_digit ** 3

    @property
    def num_neighbors(self):
        return self.neighbors.shape[-1]

    @staticmethod
    def make_onehot(grids):
        return tu.one_hot_encode(grids)[..., 1:].contiguous().float()

    def get_neighbor_embeds(self, state):
        """
        state: tensor of shape [batch_size, num_cells, hidden_vector_size]
        return: tensor of shape [batch_size, num_cells, num_neighbors, hidden_vector_size]
        """
        batch_size = state.shape[0]
        self.neighbors = self.neighbors.to(state.device)

        neighbors = tu.prepend_shape(self.neighbors, batch_size)
        state = tu.expand_along_dim(state, 1, self.num_nodes)
        return tu.select_subtensors_at(state, neighbors)

    def get_message_vectors(self, state):
        """
        state: tensor of shape [batch_size, num_cells, hidden_vector_size]
        return: tensor of shape [batch_size, num_cells, message_size]
        """
        neighbors = self.get_neighbor_embeds(state)
        state = tu.expand_along_dim(state, 2, self.num_neighbors)
        messages = self.message_linear(torch.cat([state, neighbors], dim=-1))
        return messages.sum(dim=2)

    def forward(self, grids, num_steps=16):
        device = grids.device
        batch_size = len(grids)
        outputs = []

        lstm_ch = None
        grids = grids.view(batch_size, self.num_nodes, 1)
        messages = torch.zeros(batch_size, self.num_nodes, self.message_size, device=device)
        for i in range(num_steps):
            lstm_inputs = self.update_node_linear(
                torch.cat([grids, messages], dim=-1))  # [batch_size, num_cells, hidden_vector_size]
            lstm_inputs = lstm_inputs.view(1, batch_size * self.num_nodes, self.hidden_vector_size)
            state, lstm_ch = self.update_node_lstm(lstm_inputs, lstm_ch)
            state = state.view(batch_size, self.num_nodes, self.hidden_vector_size)

            if i < num_steps - 1:  # if not the last step
                messages = self.get_message_vectors(state)

            output = self.output_linear(state)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.view(batch_size, num_steps, self.max_digit, self.max_digit, self.max_digit)
        return outputs


def get_neighbors(dim_x, dim_y, device='cpu'):
    """
    Returns a boolean tensor of shape [(dim_x * dim_y)^3, (dim_x * dim_y)^3]
    indicating whether the ith (cell, digit) is a neighbor of the jth (cell, digit)
    """
    max_digit = dim_x * dim_y
    num_cells = max_digit**2
    coords = utils.get_combinations(range(max_digit), range(max_digit))
    neighbors = np.zeros((num_cells, max_digit, num_cells, max_digit), dtype=bool)
    for i, j in itertools.product(range(num_cells), range(num_cells)):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        if x1 == x2 and y1 == y2:
            for digit in range(max_digit):
                neighbors[i, digit, j] = True
                neighbors[i, digit, j, digit] = False
        elif x1 == x2 or y1 == y2:
            for digit in range(max_digit):
                neighbors[i, digit, j, digit] = True
        elif x1 // dim_x == x2 // dim_x and y1 // dim_y == y2 // dim_y:
            for digit in range(max_digit):
                neighbors[i, digit, j, digit] = True

    return torch.tensor(neighbors, device=device).view(max_digit**3, max_digit**3)
