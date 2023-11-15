import re
import numpy as np
import random
import jsonpickle
import json
import os
import hashlib
import torch
import pandas as pd
import itertools
from collections.abc import Iterable


def kv_str(_delim=" | ", _digits=3, **kwargs):
    s = []
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if len(v.shape) == 0:
                v = v.item()
            else:
                v = v.detach().cpu()
        if isinstance(v, float):
            v = round(v, _digits)
        s.append("{}: {}".format(k, v))
    s = _delim.join(s)
    return s


def kv_print(_delim=" | ", _digits=3, **kwargs):
    """
    Pretty-prints kwargs

    :param _delim: Delimiter to separate kwargs
    :param _digits: number of decimal digits to round to
    :param kwargs: stuff to print
    :return:
    """
    print(kv_str(_delim, _digits=_digits, **kwargs))


def mkdir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def extract_args(args):
    """
    Use when *args is used as a function parameter.
    Allows both an iterable and a sequence of parameters to be passed in.
    For example, if f([1, 2, 3]) and f(1, 2, 3) will be valid inputs to the following function
        def f(*args):
            args = extract_args(args)
            ...
    @param args:
    @return:
    """
    if len(args) == 1 and is_iterable(args[0]):
        return args[0]
    return args


def is_iterable(obj, allow_str=False):
    return isinstance(obj, Iterable) and (allow_str or not isinstance(obj, str))


def short_hash(strings: list, hash_length: int):
    """
    Creates a short hash for a list of strings of a specified length.
    Raises an error if collisions are found.
    :param strings: list of strings
    :param hash_length: int
    :return:
    """
    hashes = [hashlib.sha1(s.encode("UTF-8")).hexdigest()[:hash_length] for s in strings]
    if len(set(hashes)) < len(strings):
        raise Exception("Hash collisions found. Increase hash length.")
    return hashes


# String misc
def replace(s: str, replacements: dict):
    to_replace = list(replacements)
    replace_tokens = {s: '<<<{}>>>'.format(s) for s in to_replace}
    regex_tokenize = re.compile('|'.join(map(re.escape, to_replace)))
    s = regex_tokenize.sub(lambda match: replace_tokens[match.group(0)], s)
    replace_with = {v: replacements[k] for k, v in replace_tokens.items()}
    regex_replace = re.compile('|'.join(map(re.escape, list(replace_tokens.values()))))
    return regex_replace.sub(lambda match: replace_with[match.group(0)], s)


def substring_between(s, before, after):
    i = s.index(before) + len(before)
    j = s[i:].index(after)
    return s[i:][:j]


def get_combinations(a, b):
    """
    :param a: 1-d iterable
    :param b: 1-d iterable
    :return: 1-d numpy array of (a[i], b[j]) for every i, j in a, b
    """
    return np.array(np.meshgrid(a, b)).T.reshape(-1, 2)


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def sample(s: set, n=1):
    s = random.sample(s, n)
    return s[0] if n == 1 else s


def as_dict(obj):
    return json.loads(jsonpickle.encode(obj, unpicklable=False))


def rotate(array, n):
    return array[n:] + array[:n]


def lcs_from_start(sequences):
    """
    Finds the longest common subsequence from the beginning of each sequence
    >>> lcs_from_start(['abcd', 'abcde', 'abcef'])
    'abc'
    >>> lcs_from_start(['abcd', 'bcd', 'abcdefg'])
    ''
    @param sequences:
    @return:
    """
    lss = sequences[0]
    for seq in sequences:
        while lss != seq[:len(lss)]:
            lss = lss[:-1]
    return lss


def to_dataframe(array, key_names, value_name):
    """
    Turns a multi-dimensional tensor or array into a long-format dataframe
    array: tensor or array with shape [d1, ..., dk]
    key_names: list of strings; must be same length as number of dimensions in array
    value_name: string
    """
    assert len(key_names) == len(array.shape)

    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    df = pd.DataFrame(array.reshape(-1, array.shape[-1]))
    keys = np.array(list(itertools.product(*[range(i) for i in array.shape[:-1]])))
    for i in range(len(key_names) - 1):
        key = key_names[i]
        df[key] = keys[:, i]
    df = df.melt(id_vars=key_names[:-1], var_name=key_names[-1], value_name=value_name)
    return df


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o) == set:
            return sorted(o)

        if hasattr(o, 'toJSON'):
            return o.toJSON()

        # For numpy
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(o)
        if isinstance(o, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(o)
        if isinstance(o, (np.ndarray,)):  #### This is the fix
            return o.tolist()

        # Default: return the dict of object
        return o.__dict__