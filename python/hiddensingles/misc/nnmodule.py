import torch.nn as nn


class nnModule(nn.Module):
    """
    Introduces QoL fixes to pytorch modules.
    """

    def __init__(self):
        super().__init__()
        self._properties = {}

    def __getattr__(self, item):
        if item in self.__dict__:
            return super().__getattribute__(item)
        else:
            return super().__getattr__(item)
