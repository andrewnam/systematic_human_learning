from collections.abc import Mapping
from collections import OrderedDict
import copy

_UNUSED = "__unused__"


class Object(Mapping):
    """
    A simple object that shares a similar interface to a dict, supporting functions such as
        dict(**object), list(object), object.items(), etc.
    Allows access to items through direct attribute getter and setter, e.g.
        object.hello = 'hi'
    If _default_item is defined, attempting to get an undefined attribute will yield a copy of _default_item,
        which allows for something like object.new_attr.append(19)
    Iterable access of Object always returns the same ordering
    """

    def __init__(self, _default_item=_UNUSED, **kwargs):
        self._default_item = copy.deepcopy(_default_item)
        self._items = OrderedDict(kwargs)

    def keys(self):
        for k in self._items:
            if self._items[k] != self._default_item:
                yield k

    def values(self):
        for k in self._items:
            if self._items[k] != self._default_item:
                yield self._items[k]

    def items(self):
        for k in self._items:
            if self._items[k] != self._default_item:
                yield k, self._items[k]

    def copy(self):
        """
        Returns a shallow copy over its elements
        """
        return Object(_default_item=self._default_item, **self._items)

    def map_keys(self, f):
        self._items = {f(k): v for k, v in self.items()}

    def map_values(self, f):
        self._items = {k: f(v) for k, v in self.items()}

    def map_items(self, f_keys, f_values):
        self._items = {f_keys(k): f_values(v) for k, v in self.items()}

    def __getattr__(self, item):
        if item[0] == "_":
            return super(Object, self).__getattr__(item)
        if item not in self._items:
            if self._default_item != _UNUSED:
                self._items[item] = copy.deepcopy(self._default_item)
                return self._items[item]
            else:
                raise AttributeError('{}'.format(item))
        return self._items[item]

    def __setattr__(self, key, value):
        if key[0] == "_":
            super(Object, self).__setattr__(key, value)
        else:
            self._items[key] = value

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = value

    def __iter__(self):
        for k in self._items:
            if self._items[k] is not self._default_item:
                yield k

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return dict(self._items).__repr__()
