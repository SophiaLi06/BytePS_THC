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
import torch
import math
from models.resnet import BasicBlock, BottleNeck, ResNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during pushpull')
parser.add_argument('--model', type=str, default='VGG11',
                    help='model to benchmark')
parser.add_argument('--use-bps-server', action='store_true', default=False,
                    help='Use BytePS server')
parser.add_argument('--ef', action='store_true', default=False,
                    help='use INCA compression with error feedback')
parser.add_argument('--quant-level', type=int, default=16, metavar='N',
                    help='INCA quantization levels')
parser.add_argument('--thc', action='store_true', default=False,
                    help='use THC compression during pushpull')
parser.add_argument('--new-inca', action='store_true', default=False,
                    help='use INCA (aka THC) compression during pushpull')
parser.add_argument('--new-inca-seed', type=int, default=42,
                    help='random seed for new INCA')
parser.add_argument('--overflow-freq', type=int, default=32, 
                    help='INCA overflow frequency')
parser.add_argument('--max-val', type=int, default=30,
                    help='INCA max value')
parser.add_argument('--table-dir', type=str, default='/home/byteps/byteps/torch/tables',
                    help='directory to store the tables')
parser.add_argument('--uhc', action='store_true', default=False,
                    help='use UHC compression during pushpull')
parser.add_argument('--inca', action='store_true', default=False,
                    help='use INCA (aka UHC) compression during pushpull')
parser.add_argument('--rotation', action='store_true', default=False,
                    help='use INCA compression with rotation')
### minmax for INCA - percentile
parser.add_argument('--percentile', default=1., type=float,
                    help='the percentile to use for minmax quantization')
### the maximum number of iterations if doing a partial rotation
parser.add_argument('--partial', default=1000., type=int,
                    help='the maximum number of iterations in the partial rotation')
parser.add_argument('--norm-normal', action='store_true', default=False,
                    help='use INCA compression with norm normalization')
parser.add_argument('--overflow-prob', type=float, default=0.0001, metavar='P',
                    help='per_coordinate_overflow_prob')
parser.add_argument('--dgc', action='store_true', default=False,
                    help='use DGC compression during pushpull')
parser.add_argument('--topk', action='store_true', default=False,
                    help='use TopK compression during pushpull')
parser.add_argument('--kp', type=float, default=0.1,
                        help='TopK ratio. default is 0.1.')
parser.add_argument('--terngrad', action='store_true', default=False,
                    help='use Terngrad compression during pushpull')
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

if args.model == 'VGG11' or args.model == 'ResNet18' or args.model == 'modcnn':
    train_dataset = \
        datasets.MNIST('data-%d' % bps.rank(), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.Grayscale(num_output_channels=3),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
else:
    train_dataset = \
        datasets.MNIST('data-%d' % bps.rank(), train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# BytePS: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bps.size(), rank=bps.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
print("train_loader initialized")

if args.model == 'VGG11' or args.model == 'ResNet18' or args.model == 'modcnn':
    test_dataset = \
        datasets.MNIST('data-%d' % bps.rank(), train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
else:
    test_dataset = \
        datasets.MNIST('data-%d' % bps.rank(), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
# BytePS: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=bps.size(), rank=bps.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)
print("test_loader initialized")

for data, target in train_loader:
    pass

for data, target in test_loader:
    pass

affinity = os.sched_getaffinity(0)  
# Print the result
print("Process is eligible to run on:", affinity)
# allow dpdk backend to fire up
time.sleep(20)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        print("Initializing the VGG model.")
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # The classifier below is for FEMNIST (32 / 2 / 2 / 2 / 2 / 2 = 1)
        # (the division by 2 is due to Max Pooling)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        #    nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.Flatten()]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
class ModerateCNNMNIST(nn.Module):
    def __init__(self, num_classes=10):
        print("Initializing the ModerateCNNMNIST model.")
        super(ModerateCNNMNIST, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024), # 32/2/2/2 = 4, 256 * (4*4) = 4096
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

if args.model == 'VGG11':
    model = VGG('VGG11', num_classes=10)
elif args.model == 'modcnn':
    model = ModerateCNNMNIST(num_classes=10)
elif args.model == 'ResNet18':
	# net = models.resnet18
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, color_channels=3)
else:
    model = Net()

if args.cuda:
    # Move model to GPU.
    model.cuda()

# BytePS: scale learning rate by the number of GPUs.
optimizer = optim.SGD(model.parameters(), lr=args.lr * bps.size(),
                      momentum=args.momentum)

# Find the total number of trainable parameters of the model
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if (pytorch_total_params != pytorch_total_params_trainable):
    raise Exception("pytorch_total_params != pytorch_total_params_trainable")
print("Total number of trainable parameters: " + str(pytorch_total_params_trainable))
quantization_levels = {}
for param_name, _ in model.named_parameters():
    quantization_levels[param_name] = args.quant_level
quantization_levels['batch_grads'] = args.quant_level

# BytePS: (optional) compression algorithm.
if args.thc or args.new_inca:
    compression = bps.Compression.newinca(params={'nclients': bps.get_num_worker(), 'd': pytorch_total_params_trainable, \
        'ef': args.ef, 'quantization_levels': args.quant_level, 'seed': args.new_inca_seed, \
        'overflow_frequency': args.overflow_freq, 'max_val': args.max_val, 'table_dir': args.table_dir, \
        'use_bps_server': args.use_bps_server})
elif (args.uhc or args.inca):
    compression = bps.Compression.inca(params={'nclients': bps.get_num_worker(), 'd': pytorch_total_params_trainable, \
        'ef': args.ef, 'rotation': args.rotation, 'quantization_levels': quantization_levels, \
        'partial_rotation_times': args.partial, 'percentile': args.percentile, \
        'norm_normalization': args.norm_normal, 'per_coordinate_overflow_prob': args.overflow_prob, \
        'use_bps_server': args.use_bps_server})
elif args.dgc:
    compression = bps.Compression.dgc(params={'nclients': bps.get_num_worker(),'d': pytorch_total_params_trainable, \
        'kp': args.kp, 'use_bps_server': args.use_bps_server, 'momentum':args.momentum})
elif args.topk:
    compression = bps.Compression.topk(params={'nclients': bps.get_num_worker(), 'd': pytorch_total_params_trainable, \
        'ef': args.ef, 'kp': args.kp, 'use_bps_server': args.use_bps_server})
elif args.terngrad:
    # compression = bps.Compression.terngrad(params={'d': 1000000, 'use_bps_server': args.use_bps_server, 'ef': args.ef})
    compression = bps.Compression.terngrad(params={'nclients': bps.get_num_worker(), 'd': pytorch_total_params_trainable,\
        'ef': args.ef, 'use_bps_server': args.use_bps_server})
else:
    compression = bps.Compression.fp16() if args.fp16_pushpull else bps.Compression.none()

# BytePS: wrap optimizer with DistributedOptimizer.
optimizer = bps.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)


# BytePS: broadcast parameters.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)

def train(epoch):

    model.train()
    # BytePS: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            # BytePS: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))
    


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
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

start_time = time.time()
test_time = 0.0
train_time = 0.0

for epoch in range(1, args.epochs + 1):
    if epoch > 1:
        train_start = time.time()
    train(epoch)
    if epoch > 1:
        train_time += time.time() - train_start
    if epoch > 1:
        test_start = time.time()
    test()
    if epoch > 1:
        test_time += time.time() - test_start

total_time = time.time() - start_time
print("Total Time: " + str(total_time))

