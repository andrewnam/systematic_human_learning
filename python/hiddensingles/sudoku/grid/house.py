import numpy as np
from ..misc.exceptions import InvalidEnumException
from ...misc import utils
from .coordinate import Coordinate
from enum import Enum

HouseType = Enum('HouseType', 'Row, Column, Box')


class House:

    def __init__(self, grid: np.ndarray,
                 pencil_marks: np.ndarray,
                 type: HouseType,
                 index: int,
                 dim_x=3, dim_y=3):
        self.type = type
        self.index = index
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y

        if self.type == HouseType.Row:
            self.x_min = self.x_max = index
            self.y_min = 0
            self.y_max = self.max_digit - 1
            self.array = grid[self.x_min]
            self.pencil_marks = pencil_marks[self.x_min]
        elif self.type == HouseType.Column:
            self.y_min = self.y_max = index
            self.x_min = 0
            self.x_max = self.max_digit - 1
            self.array = grid[:, self.y_min]
            self.pencil_marks = pencil_marks[:, self.y_min]
        elif self.type == HouseType.Box:
            self.x_min = (self.index // self.dim_x) * self.dim_x
            self.y_min = (self.index * self.dim_y) % self.max_digit
            self.x_max = self.x_min + self.dim_x - 1
            self.y_max = self.y_min + self.dim_y - 1
            self.array = grid[self.x_min:self.x_max + 1, self.y_min:self.y_max + 1]
            self.pencil_marks = pencil_marks[self.x_min:self.x_max + 1, self.y_min:self.y_max + 1]
        else:
            raise InvalidEnumException('type', HouseType, type)

    def as_1d(self):
        a = self.array.view()
        a.shape = self.max_digit
        return a

    def digit_pencil_marks(self, digit):
        pm = self.pencil_marks.reshape(self.max_digit, self.max_digit)
        return pm[:, digit - 1]

    def get_possible_coordinates(self, digit):
        valid_indices = self.digit_pencil_marks(digit).astype(bool)
        return np.array(list(self.get_coordinates()))[valid_indices]

    def get_coordinates(self):
        combinations = utils.get_combinations(range(self.x_min, self.x_max + 1), range(self.y_min, self.y_max + 1))
        return (Coordinate(*c) for c in combinations)

    def erase_pencil_marks(self, digit):
        self.pencil_marks[..., digit - 1] = np.zeros(self.max_digit).reshape(self.array.shape)

    def set(self, value: np.ndarray):
        assert self.array.shape == value.shape
        self.array[...] = value

    def intersect(self, other):
        """
        :param other:
        :return: A set of coordinates that the two Houses intersect on
        """
        return set(self.as_1d) & set(other.as_1d())

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, key, value):
        self.array[key] = value

    def __contains__(self, item):
        return item in self.array

    def __repr__(self):
        return "{} {}\n{}".format(self.type, self.index, self.array.__repr__())
