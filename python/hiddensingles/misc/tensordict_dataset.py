from torch.utils.data import Dataset
from .tensordict import TensorDict

class TensorDictDataset(Dataset):

    def __init__(self, tensordict: TensorDict):
        self.tensordict = tensordict

    def __len__(self):
        return self.tensordict.batch_shape[0]

    def __getitem__(self, index):
        return self.tensordict[index]
