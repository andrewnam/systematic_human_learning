from . import phase2Condition
from ..sudoku.grid.house import HouseType
from ..sudoku.grid.coordinate import Coordinate
from .hiddenSingle import HiddenSingle
from ..misc import utils


class Phase2Trial:

    def __init__(self,
                 condition: phase2Condition,
                 tutorial_goal,
                 tutorial_house_type,
                 digit_set1: set,
                 digit_set2: set):
        self.condition = condition
        digit_set = digit_set2 if self.condition.digit_set else digit_set1
        self.hidden_single = self.create_hidden_single(tutorial_goal, tutorial_house_type, digit_set)

    def create_hidden_single(self, tutorial_goal, tutorial_house_type, digit_set):
        house_type = HouseType.Column if tutorial_house_type == HouseType.Row else HouseType.Row
        house_type = house_type if self.condition.house_type else tutorial_house_type

        if house_type == HouseType.Row:
            if self.condition.house_index:
                x = utils.sample(set(range(9)) - {tutorial_goal.x}, 1)
            else:
                x = tutorial_goal.x

            if self.condition.cell_index:
                y = utils.sample(set(range(9)) - {tutorial_goal.y}, 1)
            else:
                y = tutorial_goal.y
        else:
            if self.condition.house_index:
                y = utils.sample(set(range(9)) - {tutorial_goal.y}, 1)
            else:
                y = tutorial_goal.y

            if self.condition.cell_index:
                x = utils.sample(set(range(9)) - {tutorial_goal.x}, 1)
            else:
                x = tutorial_goal.x

        goal = Coordinate(x, y)
        return HiddenSingle(goal, house_type, digit_set)

    def package(self):
        return {'condition': utils.as_dict(self.condition),
                'hidden_single': self.hidden_single.package()}