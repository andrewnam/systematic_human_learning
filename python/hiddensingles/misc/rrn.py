import torch
import torch.nn as nn
import numpy as np
import itertools


from . import torch_utils as tu
from . import utils
from .nnmodule import nnModule
from .mlp import MLP


class RRN(nnModule):
    """
    Recurrent relational network as described in Palm et al.
    """

    def __init__(self,
                 dim_x=3,
                 dim_y=3,
                 digit_embed_size=16,
                 num_mlp_layers=3,
                 hidden_vector_size=96,
                 message_size=96,
                 encode_coordinates=False):
        """
        :param dim_x: Number of cells in a Sudoku box width
        :param dim_y: Number of cells in a Sudoku box height
        :param digit_embed_size:
        :param num_mlp_layers:
        :param hidden_vector_size:
        :param message_size:
        :param encode_coordinates: If True, encodes coordinates for cell embedding. Else, just uses cell's contents.
        """
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.digit_embed_size = digit_embed_size
        self.num_mlp_layers = num_mlp_layers
        self.hidden_vector_size = hidden_vector_size
        self.message_size = message_size
        self.encode_coordinates = encode_coordinates

        self.num_embed = nn.Embedding(1 + self.max_digit, digit_embed_size)
        self.message_mlp = MLP([hidden_vector_size, hidden_vector_size],
                               [hidden_vector_size] * num_mlp_layers,
                               message_size)
        self.update_node_mlp = MLP([self.cell_embed_size, message_size],
                                   [hidden_vector_size] * num_mlp_layers,
                                   hidden_vector_size)
        self.update_node_lstm = nn.LSTM(hidden_vector_size, hidden_vector_size)
        self.output_linear = nn.Linear(hidden_vector_size, self.max_digit)

        if self.encode_coordinates:
            self.cell_embed_mlp = MLP([digit_embed_size, digit_embed_size, digit_embed_size],
                                      [hidden_vector_size] * num_mlp_layers,
                                      hidden_vector_size)
            self.grid_x, self.grid_y = torch.meshgrid(torch.arange(self.max_digit),
                                                      torch.arange(self.max_digit))

        # reusable tensors that can be pre-computed
        neighbors = get_neighbors(dim_x, dim_y)
        self.neighbors = torch.where(neighbors)[1].view((dim_x * dim_y) ** 2, -1)

    @property
    def max_digit(self):
        return self.dim_x * self.dim_y

    @property
    def num_cells(self):
        return self.max_digit ** 2

    @property
    def num_neighbors(self):
        return self.neighbors.shape[-1]

    @property
    def cell_embed_size(self):
        return self.hidden_vector_size if self.encode_coordinates else self.digit_embed_size

    def get_neighbor_embeds(self, state):
        """
        state: tensor of shape [batch_size, num_cells, hidden_vector_size]
        return: tensor of shape [batch_size, num_cells, num_neighbors, hidden_vector_size]
        """
        batch_size = state.shape[0]
        self.neighbors = self.neighbors.to(state.device)

        neighbors = tu.prepend_shape(self.neighbors, batch_size)
        state = tu.expand_along_dim(state, 1, self.num_cells)
        return tu.select_subtensors_at(state, neighbors)

    def get_message_vectors(self, state):
        """
        state: tensor of shape [batch_size, num_cells, hidden_vector_size]
        return: tensor of shape [batch_size, num_cells, message_size]
        """
        neighbors = self.get_neighbor_embeds(state)
        state = tu.expand_along_dim(state, 2, self.num_neighbors)
        messages = self.message_mlp(state, neighbors)
        return messages.sum(dim=2)

    def get_input_embedding(self, grids):
        batch_size = grids.shape[0]

        digit_embed = self.num_embed(grids).view(batch_size, self.num_cells, self.digit_embed_size)

        if self.encode_coordinates:
            self.grid_x = self.grid_x.to(grids.device)
            self.grid_y = self.grid_y.to(grids.device)

            x_embed = self.num_embed(self.grid_x)
            y_embed = self.num_embed(self.grid_y)
            x_embed = tu.prepend_shape(x_embed, batch_size).view(batch_size, self.num_cells, self.digit_embed_size)
            y_embed = tu.prepend_shape(y_embed, batch_size).view(batch_size, self.num_cells, self.digit_embed_size)
            input_embed = self.cell_embed_mlp(digit_embed, x_embed, y_embed)

        else:
            input_embed = digit_embed

        return input_embed

    def forward(self, grids, num_steps=16):
        batch_size = grids.shape[0]
        device = grids.device
        batch_size = len(grids)
        outputs = []

        input_embed = self.get_input_embedding(grids)

        lstm_ch = None
        messages = torch.zeros(batch_size, self.num_cells, self.message_size, device=device)
        for i in range(num_steps):
            lstm_inputs = self.update_node_mlp(input_embed, messages)  # [batch_size, num_cells, hidden_vector_size]
            lstm_inputs = lstm_inputs.view(1, batch_size * self.num_cells, self.hidden_vector_size)
            state, lstm_ch = self.update_node_lstm(lstm_inputs, lstm_ch)
            state = state.view(batch_size, self.num_cells, self.hidden_vector_size)

            if i < num_steps - 1:  # if not the last step
                messages = self.get_message_vectors(state)

            output = self.output_linear(state)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.view(batch_size, num_steps, self.max_digit, self.max_digit, self.max_digit)
        return outputs


def get_neighbors(dim_x, dim_y, device='cpu'):
    """
    Returns a boolean tensor of shape [(dim_x * dim_y)^2, (dim_x * dim_y)^2]
    indicating whether the ith cell is a neighbor of the jth cell
    """
    max_digit = dim_x * dim_y
    num_cells = max_digit**2
    coords = utils.get_combinations(range(max_digit), range(max_digit))
    neighbors = np.zeros((num_cells, num_cells), dtype=bool)
    for i, j in itertools.product(range(len(coords)), range(len(coords))):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        if x1 == x2 and y1 == y2:
            continue
        if x1 == x2 or y1 == y2:
            neighbors[i, j] = True
        elif x1 // dim_x == x2 // dim_x and y1 // dim_y == y2 // dim_y:
            neighbors[i, j] = True

    return torch.tensor(neighbors, device=device)
