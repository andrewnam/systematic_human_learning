import torch
from .object import Object
from collections.abc import Mapping
from collections import OrderedDict
import numpy as np
import pandas as pd
import itertools
from torch.utils.data import Dataset, DataLoader


from . import torch_utils, utils


class TensorDict(Mapping):
    """
    A custom dictionary for storing torch.Tensor objects
        May contain other TensorDicts
    Supports get/set for both item/key notation (tdict['a']) and attribute/dot notation (tdict.a).
        Set with key notation will attempt to coerce items to Tensors or TensorDicts.
            It will raise an error if that is not possible.
            For example
                >>> tdict['a'] = 4
                >>> tdict.a
                tensor(4)
        Set with dot notation will not coerce items.
            Values that are not Tensors or TensorDicts will be tracked separately.
            This is useful to store auxiliary information if necessary.
            For example
                >>> tdict.a = 4
                >>> tdict.a
                4
                >>> tdict['a']
                KeyError: 'a'
        Get with key notation only looks among tensors and TensorDicts.
        Get with dot notation will look first among Tensors and TensorDicts, then search its attributes.

        It is advised to always use the same notation for a given key or it may have unexpected results.

    All tensors must be on the same device.
    May be inherited, but super().__init__() must be called by inheriting class.
    """

    def __init__(self, **kwargs):
        if kwargs:
            devices = tuple({v.device for v in kwargs.values()})
            if len(devices) > 1:
                raise ValueError("All tensors must be on the same device. Devices found: {}".format(devices))

        self._dict = OrderedDict()
        for k, v in kwargs.items():
            self[k] = v

    @property
    def device(self):
        if self._dict:
            return next(iter(self._dict.values())).device
        else:
            return None

    @property
    def batch_shape(self):
        shapes = [t.batch_shape if isinstance(t, TensorDict) else t.shape for t in self._dict.values()]
        return utils.lcs_from_start(shapes)

    @property
    def tensors(self):
        return TensorDict(**{k: v for k, v in self._dict.items() if isinstance(v, torch.Tensor)})

    @property
    def tensordicts(self):
        return TensorDict(**{k: v for k, v in self._dict.items() if isinstance(v, TensorDict)})

    @staticmethod
    def parse_tensor(tensor, shapes: Mapping):
        """
        Splits a tensor of shape [d1, ..., dk] using shape mappings.
        Assumes that the first k-1 dimensions (d1 to d{k-1}) are batch dimensions
        @param tensor: Tensor to split
        @param shapes: Mapping (dict, Object, etc.) of {str: tuple} where each tuple is the shape of desired subtensor
        @return: TensorDict
        """
        batch_shape = tensor.shape[:-1]
        sizes = [np.prod(s) for s in shapes.values()]
        slices = torch_utils.slice(tensor, sizes)

        kwargs = {k: v for k, v in zip(shapes, slices)}
        for k, v in kwargs.items():
            kwargs[k] = v.view(*batch_shape, *shapes[k])
        return TensorDict(**kwargs)

    @staticmethod
    def cat(tensordicts: list, dim):
        if len(tensordicts) == 0:
            return TensorDict()

        tensors = {}
        for k, v in tensordicts[0]._dict.items():
            if type(v) == torch.Tensor:
                tensors[k] = torch.cat([d[k] for d in tensordicts], dim=dim)
            else:
                tensors[k] = TensorDict.cat([d[k] for d in tensordicts], dim=dim)
        return type(tensordicts[0])(**tensors)

    @staticmethod
    def stack(tensordicts: list, dim):
        if len(tensordicts) == 0:
            return TensorDict()

        tensors = {}
        for k, v in tensordicts[0]._dict.items():
            if type(v) == torch.Tensor:
                tensors[k] = torch.stack([d[k] for d in tensordicts], dim=dim)
            else:
                tensors[k] = TensorDict.stack([d[k] for d in tensordicts], dim=dim)
        return type(tensordicts[0])(**tensors)

    def map(self, f):
        tensors = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                tensors[k] = v.map(f)
            else:
                tensors[k] = f(v)
        return self.__class__(**tensors)

    def merge(self, other, overwrite=False):
        td = self.copy()
        for k, v in other.items():
            if k in self and not overwrite:
                raise KeyError("Key {} is in both TensorDicts. Set overwrite=True to allow other to overwrite.".format(k))
            td[k] = v
        return td

    def flatten(self, *keys, stack=False, dim=0):
        """
        Extracts values matching keys in lower-level TensorDicts and flattens them to a single level TensorDict
        :param keys: list of strings
        :param stack: if True, flattens by stacking along dim. Else, concatenates along dim
        :param dim: dimension to stack or cat
        :return: TensorDict
        """
        keys = utils.extract_args(keys)
        tensordict = TensorDict()
        for k in keys:
            if k in self:
                tensordict[k] = self[k]
            else:
                tds = [t.flatten(k, stack=stack, dim=dim) for t in self.tensordicts.values()]
                tds = [t for t in tds if k in t]
                tds = TensorDict.stack(tds, dim=dim) if stack else TensorDict.cat(tds, dim=dim)
                tensordict[k] = tds[k]
        return tensordict

    def round(self, n_digits):
        tensors = {}
        for k, v in self._dict.items():
            if isinstance(v, TensorDict):
                tensors[k] = v.round(n_digits)
            elif v.dtype in (torch.float, torch.double, torch.half):
                tensors[k] = (v * 10**n_digits).round() / (10**n_digits)
            else:
                tensors[k] = v
        return self.__class__(**tensors)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def copy(self):
        """
        Returns a shallow copy over its elements
        """
        return self.__class__(**self._dict)

    def clone(self):
        """
        Returns a deep copy over its elements.
        @return:
        """
        return self.map(lambda t: t.clone())

    def expand_along_dim(self, dim, n):
        """
        See torch_utils.expand_along_dim
        :param dim:
        :param n:
        :return:
        """
        td = TensorDict()
        for k, v in self.items():
            td[k] = torch_utils.expand_along_dim(v, dim, n)
        return td

    def repeat(self, dim, n, interleave=False):
        """
        Returns a TensorDict with all tensors repeated n times along dim dimension.
        If interleaved, [a b c] -> [a a b b c c].
        Else, [a b c] -> [a b c a b c].
        :param dim:
        :param n:
        :param interleave:
        :return:
        """
        td = TensorDict()
        for k, v in self.items():
            if interleave:
                v = v.repeat_interleave(n, dim=dim)
            else:
                shape = [1] * len(v.shape)
                shape[dim] = n
                v = v.repeat(shape)
            td[k] = v
        return td

    def check_device(self, item):
        """
        Checks if item can be assigned to self
        Passes either if self has no tensors/tensordicts or if devices match
        Raises an error if not valid
        @param item: torch.Tensor or TensorDict
        """
        if self.device is not None and hasattr(item, 'device') and item.device != self.device:
            raise ValueError("Tried to set Tensor on device {} to TensorDict on device {}.".format(item.device, self.device))


    def __getattr__(self, item):
        if item in self._dict:
            return self._dict[item]
        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, item))

    def __setattr__(self, key, value):
        if isinstance(value, (torch.Tensor, TensorDict)):
            self.check_device(value)
            self[key] = value
        else:
            super(TensorDict, self).__setattr__(key, value)

    def __getitem__(self, key):
        """
        If key is a string, returns the value
        If key is an index, returns a new TensorDict with index across all key-values
        """
        if isinstance(key, str):
            return self._dict[key]
        if (isinstance(key, list) or isinstance(key, tuple)) and isinstance(key[0], str):
            return TensorDict(**{k: self._dict[k] for k in key})
        return self.__class__(**{k: v[key] for k, v in self._dict.items()})

    def __setitem__(self, key, value):
        """
        If key is a string, sets the value for the key
            Attempts to coerce values to tensors or TensorDicts if possible
        If key is an index, returns a new TensorDict with index across all key-values
        """
        if isinstance(key, str): # if key is an attribute
            self.check_device(value)
            if type(value) == torch.Tensor or isinstance(value, TensorDict):
                self._dict[key] = value
            elif isinstance(value, Mapping):
                self._dict[key] = TensorDict(**value)
            else:
                self._dict[key] = torch.tensor(value, device=self.device if self.device is not None else 'cpu')
        else: # if key is a an index
            for k, v in value.items():
                dst = self._dict[k][key]
                if dst.shape != v.shape:
                    raise ValueError(
                        "Value has shape {} at key {} but target has shape {}".format(v.shape, k, dst.shape))
            for k, v in value.items():
                self._dict[k][key] = v

    def __delitem__(self, key):
        if key in self._dict:
            del self._dict[key]

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return len(self._dict)

    def repr_with_level(self, level, show_tensors):
        s = "{} with {} keys:\n".format(self.__class__.__name__, len(self))
        key_s = []
        for k, v in self._dict.items():
            if type(v) == torch.Tensor:
                key_s.append("{}: tensor with shape {}".format(k, tuple(v.shape)))
                if show_tensors:
                    key_s += ['   ' + s for s in v.__repr__().split('\n')]
            else:
                key_s.append("{}: ".format(k) + v.repr_with_level(level + 1, show_tensors))
        key_s = ['  ' * (1 + level) + s for s in key_s]
        return s + '\n'.join(key_s)

    def __repr__(self):
        return self.repr_with_level(0, False)

    def print(self, show_tensors=True, round_digits=2, detach=True, cpu=True):
        td = self
        if round_digits > 0:
            # pytorch default prints at 4, so want to set it to round_digits, then set it back
            sci_mode = torch._tensor_str.PRINT_OPTS.sci_mode
            precision = torch._tensor_str.PRINT_OPTS.precision
            torch.set_printoptions(precision=round_digits, sci_mode=False)
            td = td.round(round_digits)
        if detach:
            td = td.detach()
        if cpu:
            td = td.cpu()
        print(td.repr_with_level(0, show_tensors))

        if round_digits > 0:
            torch.set_printoptions(precision=precision, sci_mode=sci_mode)

    ### SAVING ###

    def save(self, path):
        torch.save(self.cpu().to_dict(), path)

    @staticmethod
    def load(path):
        d = torch.load(path)
        return TensorDict.from_dict(d)

    ### CASTING OPERATIONS ###

    def to_dict(self):
        d = {}
        for k, v in self.tensors.items():
            d[k] = v
        for k, v in self.tensordicts.items():
            d[k] = v.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict):
        td = TensorDict()
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                td[k] = v
            elif isinstance(v, dict):
                td[k] = TensorDict.from_dict(v)
            else:
                raise Exception("dict d contains values that are neither tensors nor dicts")
        return td

    def to_dataframe(self, dimnames=None):
        """
        Turns a flat TensorDict into a pandas DataFrame.
        Can specify names for dimensions, e.g. dimnames={0: epoch, 1: item}
        @param dimnames:
        @return:
        """
        if dimnames is None:
            dimnames = {}
        td = self.copy()

        longest_tensor = max(td, key=lambda x: len(td[x].shape))
        shape = td[longest_tensor].shape
        for k in td:
            td[k] = torch_utils.as_shape(td[k], *shape)

        td = td.numpy()
        indices = np.array(list(itertools.product(*[range(i) for i in td[longest_tensor].shape])))
        columns = {'dim_{}'.format(i): indices[:, i] for i in range(indices.shape[1])}
        for k in td:
            columns[k] = td[k].flatten()

        df = pd.DataFrame(columns)
        dimnames = {'dim_{}'.format(k): v for k, v in dimnames.items()}
        df = df.rename(dimnames, axis=1)
        return df

    def to_dataset(self):
        return TDDataset(self)

    def to_dataloader(self, **kwargs):
        return TDDataLoader(self, **kwargs)


    ### TENSOR OPERATIONS ###

    def to(self, device):
        return self.__class__(**{k: v.to(device) for k, v in self._dict.items()})

    def cpu(self):
        return self.to('cpu')

    def detach(self):
        return self.__class__(**{k: v.detach() for k, v in self._dict.items()})

    def numpy(self):
        tensors = {k: v.detach().cpu().numpy() for k, v in self._dict.items()}
        return Object(**tensors)

    def squeeze(self, dim=None):
        return self.map(lambda t: t.squeeze(dim=dim))

    def unsqueeze(self, dim):
        return self.map(lambda t: t.unsqueeze(dim=dim))

    def mean(self, dim):
        return self.map(lambda t: t.mean(dim=dim))


class TDDataset(Dataset):

    def __init__(self, tensordict: TensorDict):
        self.tensordict = tensordict

    def __len__(self):
        return self.tensordict.batch_shape[0]

    def __getitem__(self, index):
        return self.tensordict[index]


class TDDataLoader(DataLoader):

    def __init__(self, tensordict, batch_size=None, shuffle=True, **kwargs):
        if batch_size is None:
            batch_size = tensordict.batch_shape[0]
        dataset = tensordict.to_dataset()
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __iter__(self):
        iterator = super().__iter__()
        return TDDataLoaderIterator(iterator)


class TDDataLoaderIterator:

    def __init__(self, base_dataloader_iterator):
        self.base_dataloader_iterator = base_dataloader_iterator

    def __next__(self):
        dict = self.base_dataloader_iterator.__next__()
        return TensorDict(**dict)
