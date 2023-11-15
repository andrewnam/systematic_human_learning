proj_path = '/home/ajhnam/projects/hidden_singles_public/'

import sys
sys.path.append(proj_path + 'python/')

import argparse
import random
import numpy as np
import itertools
import pandas as pd
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader as DataLoader
from tqdm.auto import tqdm

from hiddensingles.misc import torch_utils as tu
from hiddensingles.misc import utils, TensorDict, TensorDictDataset, RRN


parser = argparse.ArgumentParser()
parser.add_argument("-dir", "--directory", type=str,
                    help="directory to save files to")
parser.add_argument("-n", "--name", type=str,
                    help="model name")

# Optional

# General
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-d", "--device", default='cpu',
                    help="device for model and tensors")

# Model params
parser.add_argument("-des", "--digit_embed_size", type=int, default=10,
                    help="model param: embedding size for digits")
parser.add_argument("-nml", "--num_mlp_layers", type=int, default=0,
                    help="model param: number of layers in MLP")
parser.add_argument("-hvs", "--hidden_vector_size", type=int, default=96,
                    help="model param: hidden vector size in MLP and LSTM")
parser.add_argument("-ms", "--message_size", type=int, default=96,
                    help="model param: message vector size")
parser.add_argument("-ec", "--encode_coordinates", action="store_true",
                    help="model param: whether or not cell coordinates should be encoded")
parser.add_argument("-s", "--steps", type=int, default=16,
                    help="model param: number of 'thinking' steps for the model")

# Optimizer params
parser.add_argument("-lr", "--learning_rate", type=float, default=2e-4,
                    help="optimizer param: learning rate")
parser.add_argument("-wd", "--weight_decay", type=float, default=1e-4,
                    help="optimizer param: weight decay")

# Training params
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="training param: number of epochs to train the model")
parser.add_argument("-b", "--batch_size", type=int, default=50,
                    help="training param: number of training samples per gradient update")
parser.add_argument("-ntr", "--num_train", type=int, default=0,
                    help="training param: number of training samples")
parser.add_argument("-nte", "--num_test", type=int, default=0,
                    help="number of test samples")


def get_results(model, dataset, batch_size, num_steps=16, optimizer=None):
    """
    Also runs backprop if optimizer is provided
    """
    train = optimizer is not None
    
    dataloader = DataLoader(TensorDictDataset(dataset), batch_size=batch_size, shuffle=train)
    
    losses = []
    correct = []
    for dset in dataloader:
        dset = TensorDict(**dset)
        
        if train:
            optimizer.zero_grad()
            outputs = model(dset.inputs, num_steps=num_steps)
        else:
            with torch.no_grad():
                outputs = model(dset.inputs, num_steps=num_steps)
        outputs = outputs.view(-1, num_steps, model.max_digit, model.max_digit, model.max_digit)
        targets = tu.expand_along_dim(dset.targets, 1, num_steps)
        loss = tu.cross_entropy(outputs, targets)
        
        if train:
            loss.backward()
            optimizer.step()
        
        # record
        losses.append(loss.item())
        correct.append((outputs.argmax(-1) == targets)[:,-1])
    
    correct = torch.cat(correct)
    loss = torch.tensor(losses).mean()
    accuracy = correct.float().mean().cpu()
    solved = correct.all(-1).all(-1).float().mean().cpu()
    
    results = TensorDict(loss=loss,
                         accuracy=accuracy,
                         solved=solved)
    return results


def load_data(train: bool, num_samples=0, device='cpu'):
    # loads test data if train is False
    # if num_samples is > 0, only loads that many samples
    df = pd.read_csv(proj_path + 'data/rrn/{}.csv'.format('train' if train else 'test'),
        names=['input', 'target'])
    inputs = df.input[:num_samples] if num_samples > 0 else df.input
    targets = df.target[:num_samples] if num_samples > 0 else df.target
    inputs = torch.tensor(np.array([list(s) for s in inputs], dtype=int), device=device).view(-1, 9, 9)
    targets = torch.tensor(np.array([list(s) for s in targets], dtype=int), device=device).view(-1, 9, 9) - 1
    return TensorDict(inputs=inputs, targets=targets)


def train(model, optimizer, train_dset, test_dset, num_steps, num_epochs, batch_size, verbose=True):
    tr_result = get_results(model, train_dset, batch_size=batch_size, num_steps=num_steps)
    te_result = get_results(model, test_dset, batch_size=batch_size, num_steps=num_steps)
    tr_results = [tr_result]
    te_results = [te_result]

    for epoch in tqdm(range(num_epochs)):
        tr_result = get_results(model, train_dset, batch_size=batch_size, num_steps=num_steps, optimizer=optimizer)
        tr_results.append(tr_result)
        te_result = get_results(model, test_dset, batch_size=batch_size, num_steps=num_steps)
        te_results.append(te_result)

        if verbose:
            utils.kv_print(epoch=epoch, loss=tr_result.loss,
                           tr_acc=tr_result.accuracy, tr_sol=tr_result.solved,
                           te_acc=te_result.accuracy, te_sol=te_result.solved)

    tr_results = TensorDict.stack(tr_results, 0)
    te_results = TensorDict.stack(te_results, 0)

    return tr_results, te_results


if __name__ == "__main__":
    args = parser.parse_args()

    device = int(args.device) if args.device.isdigit() else args.device
    utils.mkdir(args.directory)

    model = RRN(digit_embed_size=args.digit_embed_size,
            num_mlp_layers=args.num_mlp_layers,
            hidden_vector_size=args.hidden_vector_size,
            message_size=args.message_size,
            encode_coordinates=args.encode_coordinates).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    train_dset = load_data(train=True, num_samples=args.num_train, device=device)
    test_dset = load_data(train=False, num_samples=args.num_test, device=device)

    tr_results, te_results = train(model,
                                   optimizer,
                                   train_dset,
                                   test_dset,
                                   args.steps,
                                   args.epochs,
                                   args.batch_size,
                                   args.verbose)

    tr_df = tr_results.to_dataframe({0: 'epoch'})
    tr_df['dataset'] = 'train'
    te_df = te_results.to_dataframe({0: 'epoch'})
    te_df['dataset'] = 'test'
    df = pd.concat([tr_df, te_df])
    df['run_id'] = args.name

    torch.save(model.state_dict(), os.path.join(args.directory, 'rrn_sudoku_{}.mdl'.format(args.name)))
    df.to_csv(os.path.join(args.directory, 'rrn_sudoku_{}.tsv'.format(args.name)), sep='\t', index=False)