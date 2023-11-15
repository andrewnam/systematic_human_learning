"""
A function to load the raw data files into
"""

from tqdm.auto import tqdm
import glob
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd



def load_raw_data(dirname):
    """
    Parses raw data into a dictionary
    return:
        a dictionary of (subject_id, data)
    """
    raw_data = {}
    subject_ids = {}
    failures = []
    for filename in tqdm(sorted(glob.glob(dirname + '/*'))):
        try:
            with open(filename) as f:
                data = json.load(f)
            worker_id = os.path.basename(filename)
            if completed_hit(data):
                if worker_id not in subject_ids:
                    subject_ids[worker_id] = len(subject_ids)
                raw_data[subject_ids[worker_id]] = data
                data['worker_id'] = worker_id
        except:
            failures.append(filename)
    if failures:
        worker_ids = ['"{}"'.format(os.path.basename(f)) for f in failures]
        worker_ids = ', '.join(worker_ids)
        print("Failed to open {} files: {}".format(len(failures), worker_ids))
    return raw_data


def parse_timestring(s):
    return datetime.strptime(s, '%Y%m%d_%H%M%S%f_%Z')


def completed_hit(data):
    return bool([d for d in data['data'] if d['screen'] == 'MathBackgroundSurvey'])


def get_datum_by_type(raw_data, subject_id, key):
    data = raw_data[subject_id]['data']
    return [s for s in data if s['actionKey'] == key]


def get_datum_key_value(raw_data, subject_id, key):
    data = raw_data[subject_id]['data']
    return [s['actionValue']['value'] for s in data if s['actionKey'] == 'keyValue' and s['actionValue']['key'] == key]


def extract(lines, key, cast=None):
    def value(line):
        line = line[line.index(": ") + 2:]
        if '"' in line:
            line = line[1:]
            value = line[:line.index('"')]
        elif '}' in line:
            value = line[:line.index('}')]
        else:
            value = line
        if cast is not None:
            value = cast(value)
        return value

    lines = [value(l) for l in lines if key in l]
    return lines


def completed_experiment(data):
    return data['questionnaireResponses']['q_confidence'] != None


def get_coord_names(hidden_single_object):
    puzzle_coords = hidden_single_object['coordinates']
    coord_names = {}
    for name, coords in puzzle_coords.items():
        if type(coords) is not list:
            coords = [coords]
        for coord in coords:
            coord = (coord['x'], coord['y'])
            coord_names[coord] = name

    return coord_names


def diagnostic_test_results(raw_data):
    rows = []
    for subject_id, data in raw_data.items():
        responses = data['diagnosticTestResponses']
        solved = bool(responses and responses[-1]['correct'])
        duration = responses[-1]['responseTime'] if responses else None
        num_attempts = len(responses)
        skipped = num_attempts == 0
        row = {
            'subject_id': subject_id,
            'dtest_solved': solved,
            'dtest_duration': duration,
            'dtest_num_attempts': num_attempts,
            'dtest_skipped': skipped
        }
        rows.append(row)
    return pd.DataFrame(rows)


def get_puzzle_results(raw_data):
    rows = []
    for subject_id, data in raw_data.items():
        if not completed_experiment(data):
            continue
        records = data['puzzleRecords']
        tutorial_house = house = data['experimentDetails']['tutorial']['houseType']
        for record in records:
            attempts = record['attempts']
            num_attempts = len(record['attempts'])
            responses = ','.join([str(a['input']) for a in record['attempts']]) if record['attempts'] else "timed_out"
            house_type_cond = record['condition']['houseType']
            house = 'row' if (tutorial_house == 'row' and not house_type_cond or tutorial_house == 'column' and house_type_cond) else 'column'
            row = {
                'subject_id': subject_id,
                'phase': record['phase'],
                'trial': record['trial'],
                'house': house,
                'house_type': house_type_cond,
                'house_index': record['condition']['houseIndex'],
                'cell_index': record['condition']['cellIndex'],
                'digit_set': record['condition']['digitSet'],
                'correct': 'correct' in record and record['correct'],
                'duration': record['attempts'][-1]['responseTime'] if record['attempts'] else None,
                'timed_out': num_attempts == 0,
                'num_attempts': num_attempts,
                'responses': responses
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def get_survey_responses(raw_data):
    math_map = {
        'High school algebra': 'm_alg',
        'High school geometry': 'm_geom',
        'Trigonometric functions': 'm_trig',
        'Single-variable calculus': 'm_sv_calc',
        'Multi-variable calculus': 'm_mv_calc',
        'Linear algebra': 'm_linalg',
        'Probability & statistics': 'm_pr_stat',
        'Discrete mathematics': 'm_disc',
        'Formal logic': 'm_logic'
    }

    edu_map = {
        "Have not graduated high school": "no_hs",
        "High school graduate, diploma or equivalent": "hs",
        "Associate degree": "asso",
        "Bachelor’s degree": "bach",
        "Master’s degree": "mast",
        "Professional degree (e.g. M.D., J.D.)": "prof",
        "Doctoral degree": "phd"
    }

    rows = []
    for subject_id, data in raw_data.items():
        responses = data['surveyResponses']
        row = {
            'subject_id': subject_id,
            'gender': responses['gender'],
            'age': responses['age'],
            'education': edu_map[responses['education']],
            'edu_status': responses['degree'],
            'prior_exposure': responses['prior_exposure'],
            'prior_attempt': responses['prior_attempt'],
            'prior_completed': responses['prior_completed']
        }
        for k, v in math_map.items():
            row[v] = k in responses['MathEducation']
        rows.append(row)
    return pd.DataFrame(rows)


def get_questionnaire_responses(raw_data):
    rows = []
    for worker_id, data in raw_data.items():
        if not completed_experiment(data):
            continue
        responses = data['questionnaireResponses']

        # puzzle response
        puzzle_response = responses['q_puzzle']
        puzzle = data['experimentDetails']['questionnaire']
        puzzle_digits = puzzle['digits']
        puzzle_correct = puzzle_response == puzzle_digits['target']
        if puzzle_correct:
            puzzle_response_type = 'target'
        elif puzzle_response == puzzle_digits['distractor']:
            puzzle_response_type = 'distractor'
        elif puzzle_response in puzzle_digits['occupied']:
            puzzle_response_type = 'occupied'
        else:
            puzzle_response_type = 'other'

        # hint select
        selected_coords = responses['q_hint_select']
        if selected_coords is not None:
            selected = []
            puzzle_coordinates = get_coord_names(puzzle)
            for coord in selected_coords:
                coord = (coord['x'], coord['y'])
                if coord in puzzle_coordinates:
                    selected.append(puzzle_coordinates[coord])
                else:
                    selected.append(coord)

        row = {'subject_id': worker_id,
               'digit_target': puzzle_digits['target'],
               'digit_distractor': puzzle_digits['distractor'],
               'q_puzzle': puzzle_response_type,
               'q_puzzle_digit': puzzle_response,
               'q_hint_select': selected}

        for q_key, response in responses.items():
            if q_key in ('q_puzzle', 'q_hint_select'):
                continue
            if type(response) == dict:
                response = response['text']
            row[q_key] = response
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[['subject_id',
            'q_attn_check1',
            'q_attn_check2',
            'q_attn_check3',
            'q_puzzle',
            'q_puzzle_digit',
            'digit_target',
            'digit_distractor',
            'q_confidence',
            'q_strategy',
            'q_digit_selection',
            'q_digit_notice',
            'q_hint_select',
            'q_hint_explain',
            'q_digit_check',
            'q_check_strategy',
            'q_check_strategy_select',
            'q_additional_info'
           ]]
    return df


# get puzzle details
def map_phase_digits(raw_data, phase):
    rows = []
    for subject_id, data in raw_data.items():
        puzzles = data['experimentDetails']['phase{}'.format(phase)]
        for trial, puzzle in zip(range(len(puzzles)), puzzles):
            for digit in range(1, 10):
                if digit == puzzle['digits']['target']:
                    dtype = 'target'
                elif digit == puzzle['digits']['distractor']:
                    dtype = 'distractor'
                elif digit in puzzle['digits']['occupied']:
                    dtype = 'inhouse'
                else:
                    dtype = 'absent'
                row = {
                    'subject_id': subject_id,
                    'phase': phase,
                    'trial': 1 + trial,
                    'digit': digit,
                    'dtype': dtype
                }
                rows.append(row)
    return pd.DataFrame(rows)


def get_digit_maps(raw_data):
    phase1_digits = map_phase_digits(raw_data, 1)
    phase2_digits = map_phase_digits(raw_data, 2)
    digit_maps = phase1_digits.append(phase2_digits)
    return digit_maps


def get_first_input(responses):
    """
    Gets the first response from a string of comma-separated responses
    """
    for r in responses.split(','):
        if r.isdigit():
            return r
    return r


def get_response_types(raw_data, results):
    """
    Shows the response type of the first submitted response
    """
    digit_maps = get_digit_maps(raw_data)
    digit_maps.digit = digit_maps.digit.astype(str)
    df = results[['subject_id', 'phase', 'trial', 'responses']]
    df['digit'] = np.vectorize(get_first_input)(df.responses)
    df = df.merge(digit_maps, how='left', on=['subject_id', 'phase', 'trial', 'digit'])
    df['response_type'] = df.dtype
    df = df[['subject_id', 'phase', 'trial', 'response_type']]
    df.response_type[df.response_type.isna()] = 'blank'
    return df


def get_tutorial_house(results):
    df = results[(results.phase == 1) & (results.trial == 1)]
    df = df[['subject_id', 'house']].drop_duplicates()
    df = df.rename({'house': 'tut_house'}, axis=1)
    return df