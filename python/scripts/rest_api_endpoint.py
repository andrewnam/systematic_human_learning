from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Select current directory')
parser.add_argument('--path',
                    type=str,
                    default=None,
                    help='the path to this file')

app = Flask(__name__)
api = Api(app)
CORS(app)

@app.route('/sudoku_hs/experiment/new')
def new_experiment():
    seed = request.args.get('seed', default=None, type=str)
    return sudoku_hs_service.new_experiment(seed)


def last_directory(path):
    return os.path.basename(os.path.normpath(path))


if __name__ == '__main__':
    args = parser.parse_args()

    path = os.getcwd()
    if args.path is not None:
        if os.path.isdir(args.path) and last_directory(args.path) == 'sudoku-app':
            path = os.path.join(args.path, 'python')
        else:
            raise Exception("{} is not a valid path to sudoku-app")
    else:
        if last_directory(path) == 'scripts': # this directory
            path = os.path.join(path, '..')

    sys.path.append(path)
    from hiddensingles.experiment import sudoku_hs_service

    app.run(debug=True, host='0.0.0.0', port=5001)
