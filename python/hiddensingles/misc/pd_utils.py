import numpy as np
import pandas as pd
import itertools


def tibble(**kwargs):
    """
    replicates R's tibble function
    """
    data = np.array(list(kwargs.values())).T
    return pd.DataFrame(data, columns=kwargs.keys())


def crossing(*args, **kwargs):
    """
    replicates R's crossing / expand_grid function
    args: each arg must be a DataFrame
    kwargs: each kwarg must be an iterable
    """
    key = '_crossing_join_key'
    new_df = pd.DataFrame(list(itertools.product(*kwargs.values())),
                          columns=kwargs.keys())
    new_df[key] = 1
    for df in reversed(args):
        df[key] = 1
        new_df = pd.merge(df, new_df, on=key)
    new_df = new_df.drop(key, 1)
    return new_df
