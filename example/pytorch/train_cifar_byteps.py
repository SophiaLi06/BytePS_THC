#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import byteps.torch as bps
import time
import pickle
import datetime
from models.resnet import BasicBlock, BottleNeck, ResNet
from models.vgg import VGG
import torch.backends.cudnn as cudnn

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
# add additional options according to MXNET and pytorch imagenet
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--wd', type=float, default=0.0005,
                        help='weight decay rate. default is 0.0005.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                        help='epochs at which learning rate decays. default is 100,150.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
#########
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during pushpull')
parser.add_argument('--use-bps-server', action='store_true', default=False,
                    help='Use BytePS server')
parser.add_argument('--save-lr-accu', action='store_true', default=False,
                    help='save the learning rate and accuracy')
parser.add_argument('--ef', action='store_true', default=False,
                    help='use INCA compression with error feedback')
parser.add_argument('--quant-level', type=int, default=16, metavar='N',
                    help='INCA quantization levels')
parser.add_argument('--new-inca', action='store_true', default=False,
                    help='use INCA compression during pushpull')
parser.add_argument('--new-inca-seed', type=int, default=42,
                    help='random seed for new INCA')
parser.add_argument('--overflow-freq', type=int, default=32, 
                    help='INCA overflow frequency')
parser.add_argument('--max-val', type=int, default=30,
                    help='INCA max value')
parser.add_argument('--table-dir', type=str, default='/home/byteps/byteps/torch/tables',
                    help='directory to store the tables')
parser.add_argument('--overflow-prob', type=float, default=0.0001, metavar='P',
                    help='per_coordinate_overflow_prob')
parser.add_argument('--topk', action='store_true', default=False,
                    help='use TopK compression during pushpull')
parser.add_argument('--kp', type=float, default=0.1,
                        help='TopK ratio. default is 0.1.')
parser.add_argument('--terngrad', action='store_true', default=False,
                    help='use Terngrad compression during pushpull')
parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100'],
                    help='dataset')
parser.add_argument('--net', default='ResNet18', choices=['ResNet18', 'ResNet50', 'VGG16', 'VGG19'],
                    help='model architecture')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# BytePS: initialize library.
bps.init()
torch.manual_seed(args.seed)

if args.cuda:
    # BytePS: pin GPU to local rank.
    torch.cuda.set_device(bps.local_rank())
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 2, 'pin_memory': True, 'persistent_workers': True} if args.cuda else {}
# kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}
if args.dataset == 'CIFAR10':
	dataset = datasets.CIFAR10
	num_classes = 10
elif args.dataset == 'CIFAR100':
	dataset = datasets.CIFAR100
	num_classes = 100
train_dataset = \
    dataset('data-%d' % bps.rank(), train=True, download=True,
			transform=transforms.Compose([
				transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			]))
# Configurations from: 
# https://github.com/weiaicunzai/pytorch-cifar100/blob/11d8418f415b261e4ae3cb1ffe20d06ec95b98e4/utils.py
# train_dataset = \
#     dataset('data-%d' % bps.rank(), train=True, download=True,
# 			transform=transforms.Compose([
# 				transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(15),
# 				transforms.ToTensor(),
# 				transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
#                  (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
# 			]))

# BytePS: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bps.size(), rank=bps.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
print("train_loader initialized")

test_dataset = \
    dataset('data-%d' % bps.rank(), train=False, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
# Configurations from: 
# https://github.com/weiaicunzai/pytorch-cifar100/blob/11d8418f415b261e4ae3cb1ffe20d06ec95b98e4/utils.py
# test_dataset = \
#     dataset('data-%d' % bps.rank(), train=False, transform=transforms.Compose([
# 		transforms.ToTensor(),
# 		transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
#          (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
#     ]))
# BytePS: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bps.size(), rank=bps.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)
print("test_loader initialized")

# for data, target in train_loader:
#     pass

# for data, target in test_loader:
#     pass

print("communication backend starting")
# time.sleep(30)

affinity = os.sched_getaffinity(0)  
# Print the result
print("Process is eligible to run on:", affinity)

if args.net == 'ResNet18':
	# net = models.resnet18
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, color_channels=3)
    # model = model_factories.ResNet18(num_classes=num_classes, color_channels=3)
elif args.net == 'ResNet50':
    model = ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, color_channels=3)
    # model = model_factories.ResNet50(num_classes=num_classes, color_channels=3)
elif args.net == 'VGG16':
    model = VGG('VGG16', num_classes=num_classes)
elif args.net == 'VGG19':
    model = VGG("VGG19", num_classes=num_classes)

if args.cuda:
    # Move model to GPU.
    model.cuda()
    # model = nn.DataParallel(model, device_ids = [ 0, 1]).cuda()
    # cudnn.benchmark = True

# BytePS: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.wd)

# Find the total number of trainable parameters of the model
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if (pytorch_total_params != pytorch_total_params_trainable):
    raise Exception("pytorch_total_params != pytorch_total_params_trainable")
print("Total number of trainable parameters: " + str(pytorch_total_params_trainable))
quantization_levels = {}
for param_name, _ in model.named_parameters():
    quantization_levels[param_name] = args.quant_level
# NOTE: this is for compressing the batched gradient
quantization_levels['batch_grads'] = args.quant_level

# BytePS: (optional) compression algorithm.
if args.new_inca:
    compression = bps.Compression.newinca(params={'nclients': 1, 'd': pytorch_total_params_trainable, \
        'ef': args.ef, 'quantization_levels': args.quant_level, 'seed': args.new_inca_seed, \
        'overflow_frequency': args.overflow_freq, 'max_val': args.max_val, 'table_dir': args.table_dir, \
        'use_bps_server': args.use_bps_server})
elif args.topk:
    compression = bps.Compression.topk(params={'d': pytorch_total_params_trainable, \
        'ef': args.ef, 'kp': args.kp, 'use_bps_server': args.use_bps_server})
elif args.terngrad:
    compression = bps.Compression.terngrad(params={'d': pytorch_total_params_trainable,\
        'ef': args.ef, 'use_bps_server': args.use_bps_server})
else:
    compression = bps.Compression.fp16 if args.fp16_pushpull else bps.Compression.none

# BytePS: wrap optimizer with DistributedOptimizer.
optimizer = bps.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)
# add learning rate scheduler for Cifar100
if args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10':
    milestones = [60, 120, 160]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

# BytePS: broadcast parameters.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)
print("ready for training")

# state_dict = optimizer.state_dict()

# # Newly created optimizers will not have their state initialized, so
# # do that initialization here
# if len(state_dict['state']) == 0:
#     for group in optimizer.param_groups:
#         for p in group['params']:
#             p.grad = p.data.new(p.size()).zero_()
#     # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
#     # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
#     # forces push_pull on all model parameters, which will result in deadlock
#     # unless every rank calls step(). Therefore, to finish state initialization
#     # only call optimizer.step() with a torch.optim.Optimizer.
#     if optimizer.__module__ == bps.DistributedOptimizer.__module__:
#         super(optimizer.__class__, optimizer).step()
#     else:
#         optimizer.step()
#     state_dict = optimizer.state_dict()


computation_time = 0.0
data_move_time = 0.0
step_time = 0.0
prep_time = 0.0
batches_time = 0.0
load_time = 0.0

accuracy_and_lr = []

def train(epoch):
    global computation_time
    global step_time
    global data_move_time
    global prep_time
    global batches_time
    global load_time

    train_accuracy = 0.0

    # prep_start = torch.cuda.Event(enable_timing=True)
    # prep_end = torch.cuda.Event(enable_timing=True)
    # prep_start.record()

    model.train()
    # BytePS: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    print("train sampler set epoch")

    # prep_end.record()
    # torch.cuda.synchronize()
    # prep_time += (prep_start.elapsed_time(prep_end))

    
    load_start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()
        load_time += time.time() - load_start
        
        # time the computation time (except step function time)
        # move_start = torch.cuda.Event(enable_timing=True)
        # move_end = torch.cuda.Event(enable_timing=True)
        # move_start.record()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # move_end.record()
        # torch.cuda.synchronize()
        # data_move_time += (move_start.elapsed_time(move_end))

        # time the computation time (except step function time)
        train_start = torch.cuda.Event(enable_timing=True)
        train_end = torch.cuda.Event(enable_timing=True)
        train_start.record()

        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()
        loss = loss(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()

        train_end.record()
        torch.cuda.synchronize()
        computation_time += (train_start.elapsed_time(train_end))

        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        step_start.record()

        optimizer.step()

        step_end.record()
        torch.cuda.synchronize()
        step_time += (step_start.elapsed_time(step_end))

        pred = output.data.max(1, keepdim=True)[1]
        train_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        if batch_idx % args.log_interval == 0:
            # BytePS: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))
            print(time.time())
        
        batches_time += time.time() - batch_start

        load_start = time.time()

    train_accuracy /= len(train_sampler)
    if bps.rank() == 0:
        print('\nTrain Accuracy: {:.2f}%'.format(100. * train_accuracy))
        accuracy_and_lr.append((100. * train_accuracy))


def metric_average(val, name):
    tensor = torch.tensor(val)
    if args.cuda:
        tensor = tensor.cuda()
    avg_tensor = bps.push_pull(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    # BytePS: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)


    # BytePS: print output only on first rank.
    if bps.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
            test_loss, 100. * test_accuracy))
        if args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10':
            print('Current learning rate: '+ str(scheduler.get_last_lr()) + '\n')
            accuracy_and_lr.append((100. * test_accuracy, scheduler.get_last_lr()))
        else:
            accuracy_and_lr.append((100. * test_accuracy))



test_time = 0.0
train_time = 0.0
for epoch in range(1, args.epochs + 1):
    if epoch > 1 and (args.dataset == 'CIFAR100' or args.dataset == 'CIFAR10'):
        scheduler.step(epoch)
    if epoch == 2:
        start_time = time.time()

    train_start = time.time()
    train(epoch)
    if epoch > 1:
        train_time += time.time() - train_start
    test_start = time.time()
    test()
    if epoch > 1:
        test_time += time.time() - test_start

if args.epochs > 1:
    total_time = time.time() - start_time
    print("Total Time: " + str(total_time))

print("Total Training Time: " + str(train_time))
print("Total Testing Time: " + str(test_time))
