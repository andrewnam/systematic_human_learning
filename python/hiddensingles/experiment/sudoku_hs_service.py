import numpy as np
import random

from ..misc import utils
from ..sudoku.grid.house import HouseType
from ..sudoku.grid.coordinate import Coordinate
from .hiddenSingle import HiddenSingle
from .phase2Condition import Phase2Condition
from .phase2Trial import Phase2Trial


NUM_PHASE1_TRIALS = 25


def balanced_latin_square(n):
    numbers = list(range(n))
    random.shuffle(numbers)
    latin_square = []
    for i in range(n):
        latin_square.append(utils.rotate(numbers, (1 if i%2 else -1) * ((i+1)//2)))
    return np.array(latin_square)


def create_tutorial(digit_set):
    house_type = np.random.choice([HouseType.Row, HouseType.Column])
    x = np.random.choice(range(9))
    y = np.random.choice(range(9))
    goal = Coordinate(x, y)

    tutorial = HiddenSingle(goal, house_type, digit_set)
    return tutorial


def create_phase1(tutorial, num_phase1_trials):
    phase1 = []
    while len(phase1) < num_phase1_trials:
        hs = tutorial
        while hs.gridstrings['puzzle'] == tutorial.gridstrings['puzzle']:
            hs = HiddenSingle(tutorial.goal, tutorial.house_type, tutorial.digit_set)
        phase1.append(hs)
    return phase1


def create_phase2(tutorial, digit_set1, digit_set2):
    latin_square = balanced_latin_square(8)
    latin_square *= 2

    # Add digit_set. Each trial is encoded as a binary for ht, hi, ci, ds
    for i in range(0, 8, 2):
        indices1 = np.random.choice(range(8), 4, replace=False)
        trials1 = latin_square[i][indices1]
        trials2 = set(2 * np.arange(8)) - set(trials1)
        indices2 = np.array([np.argwhere(latin_square[i + 1] == j)[0, 0] for j in trials2])

        latin_square[i, indices1] += 1
        latin_square[i + 1, indices2] += 1

    # Sanity check
    for i in range(8):
        assert set(latin_square[i] // 2) == set(range(8))
        assert set(latin_square[:, i] // 2) == set(range(8))
    for i in range(0, 8, 2):
        assert set(latin_square[i]) | set(latin_square[i + 1]) == set(range(16))

    conditions = []
    for i, j in utils.get_combinations(range(8), range(8)):
        conditions.append(Phase2Condition(*[bool(int(i)) for i in f'{latin_square[i, j]:04b}']))
    phase2 = [Phase2Trial(c, tutorial.goal, tutorial.house_type, digit_set1, digit_set2) for c in conditions]
    return phase2


def create_questionnaire(tutorial):
    goal = tutorial.goal
    while goal == tutorial.goal:
        goal = Coordinate(np.random.choice([3, 4, 5]), np.random.choice([3, 4, 5]))
    questionnaire = HiddenSingle(goal, tutorial.house_type, tutorial.digit_set)
    return questionnaire


def create_exercises(tutorial: HiddenSingle):
    grid = tutorial.seed_grid.clone()

    mask = np.zeros((9, 9))
    if tutorial.house_type == HouseType.Row:
        mask[tutorial.goal.x] = 1
    else:
        mask[:, tutorial.goal.y] = 1

    grid.write(tutorial.coordinates['empty_single'], 2)
    grid.write(tutorial.coordinates['empty_double'], 6)
    grid.write(tutorial.coordinates['empty_box'][0], 7)
    grid.write(tutorial.coordinates['empty_box'][1], 8)
    grid.write(tutorial.coordinates['empty_box'][2], 9)

    grid.array = grid.array * mask
    gs_fullhouse = str(grid.to_gridstring().map_digits(tutorial.digits['map']))
    grid.array[tutorial.goal.x, tutorial.goal.y] = 2
    gs_contradiction = str(grid.to_gridstring().map_digits(tutorial.digits['map']))
    return gs_fullhouse, gs_contradiction


def new_experiment(seed=None):
    if seed is not None:
        seed = hash(seed) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)

    # randomly create digit sets
    digit_set1 = set(utils.sample(set(range(1, 10)), 4))
    digit_set2 = set(range(1, 10)) - digit_set1
    digit_set2 = set(utils.sample(digit_set2, 4))

    tutorial = create_tutorial(digit_set1)
    phase1 = create_phase1(tutorial, NUM_PHASE1_TRIALS)
    phase2 = create_phase2(tutorial, digit_set1, digit_set2)
    questionnaire = create_questionnaire(tutorial)
    fullhouse, contradiction = create_exercises(tutorial)

    package = {
        'digit_set1': digit_set1,
        'digit_set2': digit_set2,
        'fullhouse': fullhouse,
        'contradiction': contradiction,
        'tutorial': tutorial.package(),
        'phase1': [t.package() for t in phase1],
        'phase2': [t.package() for t in phase2],
        'questionnaire': questionnaire.package()
    }

    package = utils.UniversalEncoder().encode(package)
    return package
