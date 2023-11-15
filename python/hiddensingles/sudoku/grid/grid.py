import numpy as np
from .house import House, HouseType
from .coordinate import Coordinate
from ..misc.exceptions import InvalidWriteException
from ...misc import utils
import joblib
import re
import itertools


class Grid:

    def __init__(self, dim_x: int, dim_y: int):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.max_digit = self.dim_x * self.dim_y
        self.array = np.zeros((self.max_digit, self.max_digit), dtype=int)
        self.pencil_marks = np.ones((self.max_digit, self.max_digit, self.max_digit))
        self.rows = tuple([House(self.array, self.pencil_marks, HouseType.Row, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])
        self.columns = tuple([House(self.array, self.pencil_marks, HouseType.Column, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])
        self.boxes = tuple([House(self.array, self.pencil_marks, HouseType.Box, i, self.dim_x, self.dim_y)
                        for i in range(self.max_digit)])

    # region properties
    @property
    def num_filled_cells(self):
        return np.sum(self.array != 0)

    @property
    def num_unfilled_cells(self):
        return np.sum(self.array == 0)

    @property
    def complete(self):
        return self.num_unfilled_cells == 0

    @property
    def valid(self):
        for i in range(self.max_digit):
            k, v = np.unique(self.rows[i].array, return_counts=True)
            if not (v[k[0] == 0:] <= 1).all():
                return False
            k, v = np.unique(self.columns[i].array, return_counts=True)
            if not (v[k[0] == 0:] <= 1).all():
                return False
            k, v = np.unique(self.boxes[i].array, return_counts=True)
            if not (v[k[0] == 0:] <= 1).all():
                return False
        return True

    @property
    def solved(self):
        return self.complete and self.valid

    @property
    def is_solvable(self):
        """
        Checks if the board has any unfilled cells that have no candidates
        """
        return not np.any((self.array == 0) & (np.sum(self.pencil_marks, axis=2) == 0))

    @property
    def is_seed(self):
        return (self.array[0] == np.arange(self.array.shape[1]) + 1).all()

    # endregion

    # region static methods
    @staticmethod
    def load_array(array, dim_x, dim_y):
        assert dim_x * dim_y == array.shape[0]
        assert dim_x * dim_y == array.shape[1]
        grid = Grid(dim_x, dim_y)
        for x, y in zip(*np.nonzero(array)):
            grid.write(Coordinate(x, y), array[x, y])
        return grid

    # endregion

    def get_pencil_marks(self, digit):
        return self.pencil_marks[:,:,digit-1]

    def box_containing(self, coordinate: Coordinate):
        return self.boxes[(coordinate.x // self.dim_x) * self.dim_x + (coordinate.y // self.dim_y)]

    def to_gridstring(self):
        s = re.sub('[^0-9]', '', np.array_str(self.array))
        return GridString(self.dim_x, self.dim_y, s.replace('0', '.'))

    def clone(self):
        grid = Grid(self.dim_x, self.dim_y)
        np.copyto(grid.array, self.array)
        np.copyto(grid.pencil_marks, self.pencil_marks)
        return grid

    def set_pencil_marks(self, x, y):
        digit = self.array[x][y]
        if digit:
            self.pencil_marks[x][y] = np.zeros(self.max_digit)
            self.rows[x].erase_pencil_marks(digit)
            self.columns[y].erase_pencil_marks(digit)
            self.box_containing(Coordinate(x, y)).erase_pencil_marks(digit)

    def write(self, coordinate: Coordinate, digit):
        x = coordinate.x
        y = coordinate.y

        if self.is_candidate(x, y, digit):
            self.array[x][y] = digit
            self.set_pencil_marks(x, y)
        else:
            raise InvalidWriteException(x, y, digit, self.pencil_marks[x][y])

    def contradiction_exists(self, coordinate, digit):
        assert self[coordinate] != digit
        return (digit in self.rows[coordinate.x]) or (digit in self.columns[coordinate.y]) \
               or (digit in self.box_containing(coordinate))

    def remove(self, coordinate: Coordinate):
        digit = self[coordinate]
        assert digit > 0
        self.array[coordinate.x][coordinate.y] = 0

        # for cell at (x, y)
        f = lambda d: not self.contradiction_exists(coordinate, d)
        self.pencil_marks[coordinate.x][coordinate.y] = np.vectorize(f)(np.arange(self.max_digit) + 1)

        # for all other cells in neighborhood
        neighbors = set(self.rows[coordinate.x].get_coordinates())
        neighbors |= set(self.columns[coordinate.y].get_coordinates())
        neighbors |= set(self.box_containing(coordinate).get_coordinates())
        neighbors.remove(coordinate)

        for c in neighbors:
            self.pencil_marks[c.x][c.y][digit-1] = not self.contradiction_exists(coordinate, digit)

    def find(self, *digits):
        if len(digits) == 1:
            if type(digits[0]) == int:
                digits = [digits]
            elif len(digits[0]) > 0:
                digits = digits[0]

        coords = []
        for d in digits:
            coords += [Coordinate(x, y) for x, y in np.argwhere(self.array == d)]
        return np.array(coords)

    def map_digits(self, map):
        arr = np.array(self.array)
        for k, v in map.items():
            self.array[arr == k] = v

    # region candidates
    def is_candidate(self, x, y, digit):
        return self.pencil_marks[x][y][digit-1]

    def get_candidates(self, x, y):
        return np.nonzero(self.pencil_marks[x][y])[0] + 1

    def count_candidates(self):
        """
        :param grid:
        :return: dict (x, y) -> number of candidates at (x, y)
        """

        possibilities = np.sum(self.pencil_marks, axis=2)
        xs, ys = np.nonzero(possibilities)
        return {(x, y): possibilities[x][y] for x, y in zip(xs, ys)}
    # endregion

    # region __ functions
    def __getitem__(self, index):
        if type(index) == Coordinate:
            return self.array[index.x, index.y]
        return np.array(self.array[index])

    def __eq__(self, other):
        return np.all(self.array == other.array)

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __repr__(self):
        return self.array.__repr__()

    def __hash__(self):
        return joblib.hash(self.array).__hash__()
    # endregion


class GridString:

    def __init__(self, dim_x: int, dim_y: int, grid_string: str):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.grid_string = grid_string

    @property
    def max_digit(self):
        return self.dim_x * self.dim_y

    @property
    def num_cells(self):
        return self.max_digit**2

    @property
    def num_hints(self):
        return len(self.grid_string) - self.grid_string.count('.')

    @property
    def is_seed(self):
        return self.grid_string[:self.max_digit] == ''.join((str(i) for i in range(1, self.max_digit + 1)))

    @property
    def digit_stride(self):
        return len(str(self.max_digit))

    def get_index(self, coordinate: Coordinate):
        index = ((self.max_digit * coordinate.x) + coordinate.y) * self.digit_stride
        return index

    def get(self, coordinate: Coordinate):
        index = self.get_index(coordinate)
        return self.grid_string[index: index + self.digit_stride]

    def set(self, coordinate: Coordinate, value: int):
        index = self.get_index(coordinate)
        value = str(value).zfill(self.digit_stride)
        self.grid_string = self.grid_string[:index] + value + self.grid_string[index+self.digit_stride:]

    def clone(self):
        return GridString(self.dim_x, self.dim_y, self.grid_string)

    def traverse_grid(self):
        """
        Returns a generator that traverses through each digit in the grid
        :return: Generator of tuples (x, y, digit)
        """
        grid = self.grid_string
        for x, y in itertools.product(range(self.max_digit), range(self.max_digit)):
            if grid[0] == '.':
                yield (x, y, 0)
                grid = grid[1:]
            else:
                yield (x, y, int(grid[:self.digit_stride]))
                grid = grid[self.digit_stride:]

    def get_hints(self):
        hints = {}
        for x, y, digit in self.traverse_grid():
            if digit > 0:
                hints[Coordinate(x, y)] = digit
        return hints

    def to_grid(self):
        grid = Grid(self.dim_x, self.dim_y)
        hints = self.get_hints()

        for coord, digit in hints.items():
            grid.write(coord, digit)
        return grid

    def array(self):
        """
        Creates a numpy array of the GridString.
        This is faster than to_grid() since it doesn't 'write' the digits, thus performing no checks with pencilmarks.
        :return:
        """
        a = np.zeros((self.max_digit, self.max_digit), dtype=int)
        for x, y, digit in self.traverse_grid():
            a[x, y] = digit
        return a

    def seed_mapping(self):
        map = {self.grid_string[i]: str(i + 1) for i in range(self.max_digit)}
        return map

    def map_digits(self, map):
        map = {str(k): str(v) for k, v in map.items()}
        return GridString(self.dim_x, self.dim_y, utils.replace(self.grid_string, map))

    def make_seed(self):
        return self.map_digits(self.seed_mapping())

    @staticmethod
    def load(s: str):
        a = s.split('_')
        dim_x = int(a[0])
        dim_y = int(a[1])
        grid_string = a[2]
        return GridString(dim_x, dim_y, grid_string)

    @staticmethod
    def load_array(dim_x: int, dim_y: int, a: np.ndarray):
        s = re.sub('[^0-9]', '', np.array_str(a))
        return GridString(dim_x, dim_y, s.replace('0', '.'))

    def __eq__(self, other):
        return self.dim_x == other.dim_x and self.dim_y == other.dim_y and self.grid_string == other.grid_string

    def __lt__(self, other):
        return self.grid_string < other.grid_string

    def __hash__(self):
        return (self.dim_x, self.dim_y, self.grid_string).__hash__()

    def __repr__(self):
        return f"{self.dim_x}_{self.dim_y}_{self.grid_string}"