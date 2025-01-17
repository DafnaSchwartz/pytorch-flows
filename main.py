import argparse
import copy
import math
import sys
import ipdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sklearn.metrics as metrics
import matplotlib.pylab as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter

import datasets
import flows as fnn
import utils

if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--dataset',
    default='POWER',
    help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS')
parser.add_argument(
    '--flow', default='maf', help='flow to use: maf | realnvp | glow')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--cond',
    action='store_true',
    default=False,
    help='train class conditional flow (only for MNIST)')
parser.add_argument(
    '--eval-only',
    action='store_true',
    default=False,
    help='eval only')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=5,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1000,
    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

assert args.dataset in [
    'POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST', 'GAIT', 'PD_WRIST','PHYSIONET'
]
dataset = getattr(datasets, args.dataset)()

if args.cond:
    assert args.flow in ['maf', 'realnvp'] and args.dataset == 'MNIST', \
        'Conditional flows are implemented only for maf and MNIST'
    
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_labels = torch.from_numpy(dataset.trn.y)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_labels = torch.from_numpy(dataset.val.y)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_labels = torch.from_numpy(dataset.tst.y)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    num_cond_inputs = 10
else:
    train_tensor = torch.from_numpy(dataset.trn.x)
    train_dataset = torch.utils.data.TensorDataset(train_tensor)

    valid_tensor = torch.from_numpy(dataset.val.x)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor)

    test_tensor = torch.from_numpy(dataset.tst.x)
    test_dataset = torch.utils.data.TensorDataset(test_tensor)
    num_cond_inputs = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

num_inputs = dataset.n_dims
num_hidden = {
    'POWER': 100,
    'GAS': 100,
    'HEPMASS': 512,
    'MINIBOONE': 512,
    'BSDS300': 512,
    'MOONS': 64,
    'MNIST': 1024,
    'GAIT': 512,
    'PD_WRIST': 350,
    'PHYSIONET': 256
}[args.dataset]

act = 'tanh' if args.dataset is 'GAS' else 'relu'

modules = []

assert args.flow in ['maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
if args.flow == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu')
        ]
        mask = 1 - mask
elif args.flow == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
elif args.flow == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split-glow':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                         s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs)
        ]

model = fnn.FlowSequential(*modules)

for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.fill_(0)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

writer = SummaryWriter(comment=args.flow + "_" + args.dataset)
global_step = 0

def train(epoch):
    global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
        
        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1
        
    pbar.close()
        
    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    if args.cond:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device),
                train_loader.dataset.tensors[1].to(data.device).float())
    else:
        with torch.no_grad():
            model(train_loader.dataset.tensors[0].to(data.device))


    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, prefix='Validation', full_data=False):
    global global_step, writer

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    eval_res_list = []
    for batch_idx, data in enumerate(loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.to(device)
        with torch.no_grad():            
            res = model.log_probs(data, cond_data)
            val_loss += -res.sum().item()  # sum up batch loss
            eval_res_list.append(res.cpu().numpy())
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    if full_data:
        return np.concatenate(eval_res_list)
    return val_loss / len(loader.dataset)

def calc_performance(val_data,test_data,val_ratio):
    '''
    val data: likelihood array of the validation set
    test data: likelihood array of the test set
    val_ratio: ratio between the normal set size and the validation set size
    '''
    val_data_repeat = clear_nan(np.repeat(val_data,val_ratio))
    test_data_reshape = clear_nan(np.squeeze(test_data))
    pred = np.concatenate([test_data_reshape,val_data_repeat])
    pred = -np.clip(pred, -1000,1000)
    labels = np.concatenate([np.ones_like(test_data_reshape),np.zeros_like(val_data_repeat)])
    precision, recall, thresholds = metrics.precision_recall_curve(labels, pred)
    return precision, recall, thresholds

def clear_nan(array):
    nan_array = np.isnan(array)
    not_nan_array = ~ nan_array
    array = array[not_nan_array]
    return array
best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model

for epoch in range(args.epochs):
    if args.eval_only:
        break
    print('\nEpoch: {}'.format(epoch))

    train(epoch)
    validation_loss = validate(epoch, model, valid_loader)

    if epoch - best_validation_epoch >= 30:
        break

    if validation_loss < best_validation_loss:
        best_validation_epoch = epoch
        best_validation_loss = validation_loss
        best_model = copy.deepcopy(model)

    print(
        'Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
        format(best_validation_epoch, -best_validation_loss))

    if args.dataset == 'MOONS' and epoch % 10 == 0:
        utils.save_moons_plot(epoch, model, dataset)
    elif args.dataset == 'MNIST' and epoch % 1 == 0:
        utils.save_images(epoch, model, args.cond)

model_name = f'{args.flow}_model_scripted_{args.dataset}'
if not args.eval_only:
    # save 
    torch.save(best_model, model_name)
else:
    # load model
    best_model = torch.load(model_name)


# validate(best_validation_epoch, best_model, test_loader, prefix='Test')
test_data = validate(best_validation_epoch, best_model, test_loader, prefix='Test', full_data=True)
val_data = validate(best_validation_epoch, best_model, valid_loader, prefix='val', full_data=True)
train_data = validate(best_validation_epoch, best_model, train_loader, prefix='train', full_data=True)
precision, recall, thresholds = calc_performance(val_data=val_data,test_data=test_data,val_ratio=100)
plt.plot(recall,precision)
plt.savefig(f"recall_precision_curve_{model_name}.png")
plt.close('all')
# np.save('realnvp_test_data.npy',test_data)
# np.save('realnvp_val_data.npy',val_data)


nan_array = np.isnan(test_data)
not_nan_array = ~ nan_array
test_data = test_data[not_nan_array]
plt.hist(test_data, bins=1000, range=(-400,300),alpha=0.5, weights=np.ones(len(test_data)) / len(test_data))
plt.hist(val_data, bins=1000, range=(-400,300), alpha=0.5, weights=np.ones(len(val_data)) / len(val_data))
plt.hist(train_data, bins=1000, range=(-400,300), alpha=0.5, weights=np.ones(len(train_data)) / len(train_data))

plt.legend(["test", "validation", "train"])
plt.savefig(f"likelihood_hist_{model_name}.png")
ipdb.set_trace()


