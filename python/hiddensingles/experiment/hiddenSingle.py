import numpy as np
import random

from ..sudoku.grid.house import HouseType
from ..sudoku.grid.coordinate import Coordinate
from ..sudoku.grid.grid import Grid
from ..misc import utils


class HiddenSingle:

    def __init__(self, goal: Coordinate, house_type: HouseType, digit_set: set):
        self.house_type = house_type
        self.goal = goal
        self.digit_set = digit_set
        self.coordinates = {}
        self.gridstrings = {}
        self.digits = {}
        self.seed_grid = None
        self.grid = None

        self.create_digit_map()  # populates self.digits
        self.construct_grid()  # populates self.coordinates, self.gridstrings, self.seed_grid, and self.grid
        self.create_gridstrings()

    def create_digit_map(self):
        target, distractor = utils.sample(self.digit_set, 2)
        s = list(set(range(1, 10)) - {target, distractor})
        np.random.shuffle(s)
        s = {i + 3: s[i] for i in range(0, 7)}
        s[1] = target
        s[2] = distractor

        self.digits = {
            'target': target,
            'distractor': distractor,
            'occupied': [s[3], s[4], s[5]],
            'digit_set': self.digit_set,
            'map': s
        }

    def construct_grid(self):
        goal = self.goal if self.house_type == HouseType.Row else self.goal.T

        grid = Grid(3, 3)
        grid.write(goal, 1)  # write target cell for pencilmarks

        target_box = grid.box_containing(goal)
        band_min = (target_box.index // 3) * 3
        band_indices = set(range(band_min, band_min + 3))

        row_coords = set(grid.rows[goal.x].get_coordinates())

        # box constraint + box distractor
        remaining_boxes = band_indices - {target_box.index}
        box_constraint_index = utils.sample(remaining_boxes)
        constrained_box = grid.boxes[box_constraint_index]
        constrained_box_coords = set(constrained_box.get_coordinates())

        available_coords = constrained_box_coords - row_coords
        constraint_coord, distractor_coord = utils.sample(available_coords, 2)
        grid.write(constraint_coord, 1)
        grid.write(distractor_coord, 2)

        self.coordinates["empty_box"] = sorted(constrained_box_coords & row_coords)
        self.coordinates["target_box"] = constraint_coord
        self.coordinates["distractor_box"] = distractor_coord

        # 3 occupied constraints
        available_coords = row_coords - constrained_box_coords - {goal}
        constraint_coords = utils.sample(available_coords, 3)
        for i, c in zip(range(3, 6), constraint_coords):
            grid.write(c, i)
        self.coordinates["occupied"] = constraint_coords

        # 2 column constraints, 1 column distractor, 1 random distractor
        available_coords -= set(constraint_coords)
        band_coordinates = set(utils.flatten([grid.boxes[i].get_coordinates() for i in band_indices]))
        coord1, coord2 = random.sample(available_coords, 2)  # for random ordering, col1 has distractor, col2 does not

        # column constraint 1 & distractor
        column = grid.columns[coord1.y]
        constraint_coord, distractor_coord = utils.sample(set(column.get_coordinates()) - band_coordinates, 2)
        grid.write(constraint_coord, 1)
        grid.write(distractor_coord, 2)
        self.coordinates["empty_double"] = Coordinate(goal.x, coord1.y)
        self.coordinates["target_double"] = constraint_coord
        self.coordinates["distractor_double"] = distractor_coord

        # column constraint 2
        column = grid.columns[coord2.y]
        # avoid row conflict with first column constraint
        constraint_coord = utils.sample(set(column.get_possible_coordinates(1)) - band_coordinates)
        grid.write(constraint_coord, 1)
        self.coordinates["empty_single"] = Coordinate(goal.x, constraint_coord.y)
        self.coordinates["target_single"] = constraint_coord

        # distractor 2
        mask = np.ones((9, 9), dtype=bool)
        band_min_row = (goal.x // 3) * 3
        mask[band_min_row:band_min_row + 3] = 0
        mask[:, coord2.y] = 0
        mask[:, goal.y] = 0  # so that it doesn't block the target cell
        valid_x, valid_y = np.where(grid.get_pencil_marks(2).astype(bool) & mask)
        valid_i = np.random.randint(len(valid_x))
        valid_coord = Coordinate(valid_x[valid_i], valid_y[valid_i])
        grid.write(valid_coord, 2)
        self.coordinates["distractor_single"] = valid_coord

        grid.remove(goal)

        if self.house_type == HouseType.Column:
            grid = Grid.load_array(grid.array.T, 3, 3)
            for coord_type in self.coordinates.keys():
                if type(self.coordinates[coord_type]) == Coordinate:
                    self.coordinates[coord_type] = self.coordinates[coord_type].T
                elif type(self.coordinates[coord_type]) == list:
                    self.coordinates[coord_type] = [c.T for c in self.coordinates[coord_type]]

        self.seed_grid = grid
        self.grid = self.seed_grid.clone()
        self.grid.map_digits(self.digits['map'])
        self.coordinates['goal'] = self.goal

    def create_gridstrings(self):
        puzzle_seed = self.seed_grid.to_gridstring()
        puzzle = puzzle_seed.map_digits(self.digits['map'])
        solution_seed = puzzle_seed.clone()
        solution_seed.set(self.goal, 1)
        solution = solution_seed.map_digits(self.digits['map'])

        self.gridstrings = {
            'puzzle': str(puzzle),
            'solution': str(solution),
            'puzzle_seed': str(puzzle_seed),
            'solution_seed': str(solution_seed)
        }

    def package(self):
        return {
            'house_type': self.house_type.name,
            'digits': self.digits,
            'coordinates': self.coordinates,
            'gridstrings': self.gridstrings
        }

    def __repr__(self):
        return str(self.grid)
