from . import utils
import numpy as np
import math
import re
import torch
import torch.nn.functional as F


def with_eval(func):
    """
    A decorator for any function that is meant for running a nn.Module for evaluation.
    Automatically applies torch.no_grad() and model.eval().
    """
    @torch.no_grad()
    def wrapper(model, *args, **kwargs):
        training = model.training
        model.eval()
        retval = func(model, *args, **kwargs)
        if training:
            model.train()
        return retval

    return wrapper


def as_shape(tensor, *shape, reversed=False):
    shape = utils.extract_args(shape)
    if reversed:
        subshape = shape[-len(tensor.shape):]
        dim = 0
    else:
        subshape = shape[:len(tensor.shape)]
        dim = -1

    for ea, eb in zip(subshape, tensor.shape):
        assert ea == eb or ea == 1 or eb == 1
    while len(tensor.shape) < len(shape):
        tensor = tensor.unsqueeze(dim)

    return tensor.expand(shape)


def batch_arange(*shape, device='cpu'):
    """
    Returns a tensor of batch_shape where each vector in the last dimension is range(shape[-1])
    @param batch_shape:
    @return:
    """
    shape = utils.extract_args(shape)
    return as_shape(torch.arange(shape[-1], device=device), shape, reversed=True)


def bcast_mul(a, b):
    """
    Broadcasts element-wise multiplication between similarly shaped tensors
    Specifically, shape of a is a subsequence of b's shape (see match_shape)
    e.g. a.shape is [3, 4, 6] and b.shape is [3, 4, 5, 6, 7]
    """
    long = a if len(a.shape) > len(b.shape) else b
    return match_shape(a, b) * long


def cross_entropy(input, target, reduction='mean'):
    """
    Wrapper for cross_entropy loss so that the dimensions are more intuitive.
    Permutes the dimensions so that the last dim of input corresponds to number of classes.
    """
    dims = [0, -1] + list(range(1, len(input.shape) - 1))
    input = input.permute(dims)
    return F.cross_entropy(input, target, reduction=reduction)


def ehdci(samples, p=.95):
    """
    Computes the empirical highest density continuous interval, applied along the last dim of samples
    :param samples: tensor of shape [d1, ..., dn]
    :param p: fraction of samples to include in hdci
    :return: tensor of shape [d1, ..., d{n-1}, 2] where the last dimension contains HDCI_low and HDCI_high
    """
    interval_size = math.ceil(p * samples.shape[-1]) # number of samples in hdci
    samples = samples.sort(-1)[0]
    intervals = samples.unfold(-1, interval_size, 1) # creates windows of interval_size along last dim
    ranges = intervals[..., -1] - intervals[..., 0]
    idx = ranges.argmin(-1)
    hdci = select_subtensors(intervals, idx)
    return torch.stack([hdci[..., 0], hdci[..., -1]], dim=-1)


def expand_along_dim(tensor, dim, n):
    """
    Expands a tensor n times along dimension dim.
    """
    tensor = tensor.unsqueeze(dim)
    shape = list(tensor.shape)
    shape[dim] = n
    return tensor.expand(*shape)


def extend_mul(a, b):
    """
    a: tensor of shape [a_1, ..., a_m, d]
    b: tensor of shape [d, b_1, ..., b_n]

    Multiplies a and b element wise along d

    returns tensor of shape [a_1, ..., a_m, d, b_1, ..., b_n]
    """
    b = prepend_shape(b, a.shape[:-1])
    return bcast_mul(a, b)


def logit(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=float)
    return torch.log(x) - torch.log1p(-x)


def make_tensor(x, dtype=None, device='cpu'):
    """
    Allows manual construction of a tensor from individual tensors while allowing for backprop
    to the individual input tensors.

    x: list array. May contain multiple dimensions, tensors, floats, ints, etc.
    make_tensor needs to be called after each optimizer step
    Creating it once and running backward on it multiple times will not work.

    Note that this is not an efficient method. For a tensor with N dims, each value will need to be copied N times.

    Example Usage:
    * x1 and x2 are scalars to optimize
    * x3 and x4 should not be optimized
    * x5 is a vector to optimize
    * Note that make_tensor is called inside the loop

        x1 = torch.tensor(4., requires_grad=True)
        x2 = torch.tensor(3., requires_grad=True)
        x3 = 2
        x4 = 3
        x5 = torch.tensor([3., 2], requires_grad=True)

        _x = [[x1, x2], [x3, x4], x5]
        optimizer = optim.Adam([x1, x2, x5])
        for i in tqdm(range(1000)):
            x = make_tensor(_x, dtype=torch.float32)
            optimizer.zero_grad()
            y = x.pow(2).sum()
            loss = F.mse_loss(y, torch.zeros_like(y))
            loss.backward()
            optimizer.step()
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, int) or isinstance(x, float):
        return torch.tensor(x, dtype=dtype, device=device)
    return torch.stack([make_tensor(xi, dtype, device) for xi in x], dim=0)


def match_shape(a, b, reversed=False):
    """
    Makes the shape of the shorter tensor match the shape of the longer tensor by expanding on appropriate dimensions
    Assumes that shape of shorter is a subsequence of longer's shape
        e.g. a.shape is [3, 4, 5] and b.shape is [3, 4, 5, 6, 7]
    A '1' can be used as a placeholder for the shorter tensor
        e.g. a.shape is [3, 1, 5] and b.shape is [3, 4, 5, 6, 7]
    """
    if len(a.shape) > len(b.shape):
        short = b
        long = a
    else:
        short = a
        long = b
    return as_shape(short, *long.shape, reversed=reversed)


def nll_loss(input, target, reduction='mean'):
    """
    Wrapper for F.nll_loss so that the dimensions are more intuitive.
    Permutes the dimensions so that the last dim of input corresponds to number of classes.
    """
    dims = [0, -1] + list(range(1, len(input.shape) - 1))
    input = input.permute(dims)
    return F.nll_loss(input, target, reduction=reduction)


def normalize(x, dim=-1):
    """
    Divides every element by the sum across the dim.
    :param x:
    :param dim:
    :return:

    >>> normalize(torch.rand(4, 4), dim=1).sum(1)
    tensor([1., 1., 1., 1.])
    """
    return x / x.sum(dim).unsqueeze(dim)


def one_hot_encode(x, num_items=None):
    num_items = int(torch.max(x)+1) if num_items is None else num_items
    encoding = torch.zeros(x.shape + (num_items, ))
    if x.is_cuda:
        encoding = encoding.cuda(x.get_device())
    dim = len(x.shape)
    x = x.view(x.shape + (1, ))
    return encoding.scatter_(dim, x, 1).type(torch.long)


def prepend_shape(tensor, *shape):
    """
    Expands a tensor along dimensions 0, ..., k-1 so that it has shape [*shape, tensor.shape]
    Inputs:
        tensor: any tensor
        *shape: ints describing lengths of prepended dimensions 0, ..., k-1
    Outputs:
        tensor of shape [shape[0], ..., shape[k-1], tensor.shape[0], ..., tensor.shape[-1]]
    """
    shape = utils.extract_args(shape)
    new_shape = tuple(shape) + tensor.shape
    return as_shape(tensor, *new_shape, reversed=True)


def round(x, decimals=0, keep_device=True):
    """
    This is does not retain gradient information (which would be useless anyway)
    :param x: tensor
    :param decimals: number of decimals places to round
    :param keep_device: if True, will return a tensor in the same device
    :return:
    """
    device = x.device
    x = x.detach().cpu().numpy().round(decimals)
    x = torch.tensor(x, device=device if keep_device else 'cpu')
    return x


def select(tensor, indices, select_dims=0):
    """
    Selects subtensors from tensor using indices. Allows advanced indexing.

    select elements from last dim
    >>> tensor = torch.arange(2 * 3 * 4 * 5).view(2, 3, 4, 5)
    >>> indices = torch.tensor([[1, 2, 3], [0, 0, 0]])
    >>> target = torch.tensor([33, 60])
    >>> selected = select(tensor, indices)
    >>> assert (selected == target).all()
    """
    if len(indices.shape) == 1:
        indices = indices.unsqueeze(1)

    select_tensor_dim = select_dims + 1
    while indices.numel() > 0:
        last_indices = match_shape(indices[..., -1:], tensor)
        tensor = tensor.gather(-select_tensor_dim, last_indices)  # [..., 0]
        tensor = swap_dims(tensor, -select_tensor_dim, -1)[..., 0]
        if select_tensor_dim > 2:
            tensor = swap_dims(tensor, -(select_dims), -1)
        indices = indices[..., :-1]
    return tensor


def select_subtensors(tensor, indices):
    """
    tensor: tensor with shape [d_1, ..., d_k, ..., d_n]
    indices: tensor with shape [d_1, ..., d_k]
    output: tensor with shape [d_1, ..., d_(k-1), d_(k+1), ..., d_n]
    >>> a = torch.arange(2*3*4*5).view(2,3,5,4)
    >>> b = torch.tensor([[0, 0, 0], [3,2,1]])
    >>> select_subtensors(a, b)
    tensor([[[  0,   1,   2,   3],
             [ 20,  21,  22,  23],
             [ 40,  41,  42,  43]],

            [[ 72,  73,  74,  75],
             [ 88,  89,  90,  91],
             [104, 105, 106, 107]]])
    """
    dim = len(indices.shape)
    length = tensor.shape[dim]
    indices = indices + (indices < 0)*length # gather throws CUDA error with negative indices
    while len(indices.shape) < len(tensor.shape):
        indices = indices.unsqueeze(-1)
    shape = (-1,)*(dim+1) + tensor.shape[dim+1:]
    indices = indices.expand(shape)
    return tensor.gather(dim, indices).squeeze(dim)


def select_subtensors_at(tensor, indices):
    """
    Similar to select_subtensors, except that the last dim of indices doesn't have to match with tensor.

    @param tensor: tensor with shape [d_1, ..., d_k, ..., d_n]
    @param indices: tensor with shape [d_1, ..., d_k, m]
    @return: tensor with shape [d_1, ..., d_k, m, d_(k+2), ..., d_n]
    >>> a = torch.arange(2*3*4*5).view(2,3,4,5)
    >>> b = torch.tensor([[2], [1]])
    >>> c = select_subtensors_at(a, b)
    >>> assert (c[0] == a[0, 2]).all()
    >>> assert (c[1] == a[1, 1]).all()
    """
    k = len(indices.shape) - 1
    m = indices.shape[-1]
    tensor = expand_along_dim(tensor, k, m)
    return select_subtensors(tensor, indices)


def slice(tensor, sizes):
    """
    Slices a tensor along its last dimension to tensors of shape [..., size]
    e.g. slice(torch.arange(10), [1,2,3,4]) => [0], [1, 2], [3, 4, 5], [6, 7, 8, 9]
    Requires that the length of last dim of tensor equals the sum of sizes
    :param tensor:
    :param sizes:
    :return:
    """
    assert sum(sizes) == tensor.shape[-1]
    slices = []
    i = 0
    for s in sizes:
        slices.append(tensor[..., i:i+s])
        i += s
    return slices


def swap_dims(tensor, dim_a, dim_b):
    """
    Swaps two dimensions of a tensor
    if tensor has shape [1, 2, 3, 4], dim_a = 1, dim_b = 3
    output tensor has shape [1, 4, 3, 2]
    """
    shape = torch.arange(len(tensor.shape))
    shape[dim_a], shape[dim_b] = dim_b, dim_a
    return tensor.permute(*shape)


def to_strings(x, dims=1, sep=','):
    """
    Turns the last dims dimensions into a string. Returns a numpy array.
    :param x: tensor or numpy array
    :param dims:
    :param sep:
    :return:
    """
    shape = x.shape[:-dims]

    if isinstance(x, torch.Tensor):
        x = x.view(-1, *x.shape[-dims:]).numpy()
    else:
        x = x.reshape(-1, *x.shape[-dims:])
    x = [np.array2string(a, edgeitems=np.inf, separator=sep) for a in x]
    x = [re.sub(r'[\ \[\]\n]|', '', s) for s in x]  # .split(',\n')
    return np.array(x).reshape(shape)


def write_subtensors(tensor, indices, values, clone=False):
    """
    Writes values to specific indices of tensor along dim k.
    Indices indicate which subtensors of shape [d{k+1}, ..., dn] to replace
    using value tensors of shape [d{k+1}, ..., dn]
    @param tensor: tensor with shape [d1, ..., dk, ..., dn]
    @param indices: tensor with shape [d1, ..., d{k-1}]
    @param values: tensor with shape [d1, ..., d{k-1}, d{k+1}, ..., dn]
    @param clone: whether or not to return a copy of tensor
        If False, will write directly to the tensor and not return a new tensor
    @return: None if clone is False. The new tensor if clone is True.

    Example 1: Writing scalars to 2D matrix
    >>> tensor = torch.arange(2*3).view(2,3)
    >>> indices = torch.tensor([1, 2])
    >>> values = torch.tensor([100, 200])
    >>> write_subtensors(tensor, indices, values)
    >>> tensor
    tensor([[  0, 100,   2],
            [  3,   4, 200]])

    Example 2: Writing scalars to 3D tensor
    >>> tensor = torch.arange(2*3*4).view(2,3,4)
    >>> indices = torch.tensor([[1, 2, 3], [0, 0, 0]])
    >>> values = torch.tensor([[100, 200, 300], [400, 500, 600]])
    >>> write_subtensors(tensor, indices, values)
    >>> tensor
    tensor([[[  0, 100,   2,   3],
             [  4,   5, 200,   7],
             [  8,   9,  10, 300]],

            [[400,  13,  14,  15],
             [500,  17,  18,  19],
             [600,  21,  22,  23]]])

    Example 3: Writing vectors to 4D tensor
    >>> tensor = torch.arange(2*3*4*2).view(2,3,4,2)
    >>> indices = torch.tensor([[1, 2, 3],
                                [0, 0, 0]])
    >>> values = torch.tensor([[[100, 100],
                                [200, 200],
                                [300, 300]],
                               [[400, 400],
                                [500, 500],
                                [600, 600]]])
    >>> write_subtensors(tensor, indices, values)
    >>> tensor
    tensor([[[[  0,   1],
              [100, 100],
              [  4,   5],
              [  6,   7]],

             [[  8,   9],
              [ 10,  11],
              [200, 200],
              [ 14,  15]],

             [[ 16,  17],
              [ 18,  19],
              [ 20,  21],
              [300, 300]]],


            [[[400, 400],
              [ 26,  27],
              [ 28,  29],
              [ 30,  31]],

             [[500, 500],
              [ 34,  35],
              [ 36,  37],
              [ 38,  39]],

             [[600, 600],
              [ 42,  43],
              [ 44,  45],
              [ 46,  47]]]])
    """
    if clone:
        tensor = tensor.clone()
    dim = len(indices.shape)
    values = expand_along_dim(values, dim, tensor.shape[dim])
    indices = match_shape(indices, tensor)
    tensor.scatter_(dim, indices, values)
    if clone:
        return tensor


def bmm(a, b):
    """
    Matrix multiplies last 2 dimensions while preserving the shape of the other dimensions
    :param a: tensor of shape [d1, ..., dk, x, y]
    :param b: tensor of shape [d1, ..., dk, y, z]
    :return: tensor of shape [d1, ..., dk, x, z]
    """
    batch_shape = a.shape[:-2]
    a = a.view(-1, *a.shape[-2:])
    b = b.view(-1, *b.shape[-2:])
    c = a.bmm(b)
    c = c.view(*batch_shape, *c.shape[-2:])
    return c


def batch_outer(a, b, op='mul'):
    """
    Takes tensor a of shape [B, M] and tensor b of shape [B, N]
    and returns a batchwise outer-product tensor of shape [B, M, N]
    Can provide op to perform add, sub, mul, div
    """
    a = a.unsqueeze(-1)
    b = b.unsqueeze(-2)
    if op == 'add':
        return a+b
    if op == 'sub':
        return a-b
    if op == 'mul':
        return a*b
    if op == 'div':
        return a/b
    raise Exception("op must be one of 'add', 'sub', 'mul', or 'div'")


def diag_sum(x, direction='dr'):
    """
    :param x: tensor of shape [batch_size, d, d]
    :param direction:
        'dr': computes diagonally down-right, going from top-left to bottom-right
        'dl': computes diagonally down-left, going from top-right to bottom-left
        'ur': computes diagonally up-right, going from bottom-left to top-right
        'ul': computes diagonally up-left, going from bottom-right to top-left
    :return:
        tensor of shape [batch_size, 2*d - 1]

    >>> x = torch.arange(18).view(2, 3, 3)
    >>> x
    tensor([[[ 0,  1,  2],
             [ 3,  4,  5],
             [ 6,  7,  8]],

            [[ 9, 10, 11],
             [12, 13, 14],
             [15, 16, 17]]])
    >>> diag_sum(x)
    tensor([[ 0,  4, 12, 12,  8],
            [ 9, 22, 39, 30, 17]])
    """
    assert direction in ('dr', 'dl', 'ur', 'ul')
    #     no transform -> ur
    if direction == 'dl':
        x = x.flip(-2).flip(-1)
    elif direction == 'dr':
        x = x.flip(-2)
    elif direction == 'ul':
        x = x.flip(-1)
    dim = x.shape[-1]
    num_diagonls = 2 * dim - 1
    w = torch.eye(dim, dtype=x.dtype, device=x.device)
    w = prepend_shape(w, 1, 1)
    x = x.unsqueeze(1)

    result = F.conv2d(x, w, padding=num_diagonls // 2)[:, 0, num_diagonls // 2]
    return result
