from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import byteps.torch as bps
import tensorboardX
import os
import math
from tqdm import tqdm
import pickle
import datetime
import time
# from models.vgg import VGG
import atexit
import random

# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during pushpull')
parser.add_argument('--batches-per-pushpull', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing pushpull across workers; it multiplies '
                         'total batch size.')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
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
parser.add_argument('--errors-file', type=str, default=None)
parser.add_argument('--overflow-prob', type=float, default=0.0001, metavar='P',
                    help='per_coordinate_overflow_prob')
parser.add_argument('--dummy', action='store_true', default=False,
                    help='use Dummy compression during pushpull')
parser.add_argument('--dgc', action='store_true', default=False,
                    help='use DGC compression during pushpull')
parser.add_argument('--topk', action='store_true', default=False,
                    help='use TopK compression during pushpull')
parser.add_argument('--kp', type=float, default=0.1,
                        help='TopK ratio. default is 0.1.')
parser.add_argument('--terngrad', action='store_true', default=False,
                    help='use Terngrad compression during pushpull')
parser.add_argument('--net', default='VGG16', choices=['VGG16', 'VGG19'],
                    help='model architecture')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

pushpull_batch_size = args.batch_size * args.batches_per_pushpull

bps.init()
torch.manual_seed(args.seed)

if args.cuda:
    # BytePS: pin GPU to local rank.
    torch.cuda.set_device(bps.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# BytePS: broadcast resume_from_epoch from rank 0 (which will have
# checkpoints) to other ranks.
#resume_from_epoch = bps.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
#                                  name='resume_from_epoch').item()

# BytePS: print logs on the first worker.
verbose = 1 if bps.rank() == 0 else 0

# BytePS: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if bps.rank() == 0 else None


kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers': True} if args.cuda else {}
train_dataset = \
    datasets.ImageFolder(args.train_dir,
                         transform=transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
# BytePS: use DistributedSampler to partition data among workers. Manually specify
# `num_replicas=bps.size()` and `rank=bps.rank()`.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=bps.size(), rank=bps.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=pushpull_batch_size,
    sampler=train_sampler, **kwargs)

val_dataset = \
    datasets.ImageFolder(args.val_dir,
                         transform=transforms.Compose([
                             transforms.Resize(256),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                         ]))
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=bps.size(), rank=bps.rank())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)

print("communication backend starting")
time.sleep(30)

# Set up standard VGG16 model.
# model = VGG('VGG16', num_classes=1000)
if args.net == 'VGG16':
    model = models.vgg16()
elif args.net == 'VGG19':
    model = models.vgg19()
# decrease dropout rate as recommended in the Terngrad Paper
if args.terngrad:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.2
    if args.base_lr == 0.0125:
        print("update base_lr to 0.00125 for TernGrad")
        args.base_lr == 0.00125

if args.cuda:
    # Move model to GPU.
    model.cuda()

# BytePS: scale learning rate by the number of GPUs.
# Gradient Accumulation: scale learning rate by batches_per_pushpull
optimizer = optim.SGD(model.parameters(), 
                    #   lr=args.base_lr * args.batches_per_pushpull,
                      lr=(args.base_lr *
                          args.batches_per_pushpull * bps.size()),
                      momentum=args.momentum, weight_decay=args.wd)

# Find the total number of trainable parameters of the model
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
if (pytorch_total_params != pytorch_total_params_trainable):
    raise Exception("pytorch_total_params != pytorch_total_params_trainable")
print("Total number of trainable parameters: " + str(pytorch_total_params_trainable))
quantization_levels = {}
for param_name, _ in model.named_parameters():
    quantization_levels[param_name] = 16
# NOTE: this is for compressing the batched gradient
quantization_levels['batch_grads'] = 16
compressor_name = 'none'
# BytePS: (optional) compression algorithm.
if args.new_inca:
    compression = bps.Compression.newinca(params={'nclients': 1, 'd': pytorch_total_params_trainable, \
        'ef': args.ef, 'quantization_levels': args.quant_level, 'seed': args.new_inca_seed, \
        'overflow_frequency': args.overflow_freq, 'max_val': args.max_val, 'table_dir': args.table_dir, \
        'use_bps_server': args.use_bps_server})
    if args.errors_file:
        with open(args.errors_file, 'rb') as pkl_file:
            compression.errors = pickle.load(pkl_file)
    compressor_name = 'THC'
elif args.inca:
    compression = bps.Compression.inca(params={'nclients': 1, 'd': pytorch_total_params_trainable, \
        'ef': args.ef, 'rotation': args.rotation, 'quantization_levels': quantization_levels, \
        'partial_rotation_times': args.partial, 'percentile': args.percentile, \
        'norm_normalization': args.norm_normal, 'per_coordinate_overflow_prob': args.overflow_prob, \
        'use_bps_server': args.use_bps_server})
    compressor_name = 'UHC'
elif args.dummy:
    compression = bps.Compression.dummy(params={'modify_idx': -1})
    compressor_name = 'dummy'
elif args.dgc:
    compression = bps.Compression.dgc(params={'d': pytorch_total_params_trainable, \
        'kp': args.kp, 'use_bps_server': args.use_bps_server, 'momentum':0.0})
    compressor_name = "dgc"
elif args.topk:
    compression = bps.Compression.topk(params={'d': pytorch_total_params_trainable, \
        'ef': args.ef, 'kp': args.kp, 'use_bps_server': args.use_bps_server})
    compressor_name = 'topk'
elif args.terngrad:
    compression = bps.Compression.terngrad(params={'d': pytorch_total_params_trainable, \
        'use_bps_server': args.use_bps_server})
    compressor_name = 'terngrad'
else:
    compression = bps.Compression.fp16() if args.fp16_pushpull else bps.Compression.none()

# BytePS: wrap optimizer with DistributedOptimizer.
optimizer = bps.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters(),
    compression=compression,
    backward_passes_per_step=args.batches_per_pushpull)

# Restore from a previous checkpoint, if initial_epoch is specified.
# BytePS: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and bps.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# BytePS: broadcast parameters & optimizer state.
bps.broadcast_parameters(model.state_dict(), root_rank=0)
bps.broadcast_optimizer_state(optimizer, root_rank=0)

accuracy_and_lr = []
throughput_log = []

computation_time = 0.0
step_time = 0.0

def train(epoch):
    global computation_time
    global step_time

    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    train_accuracy_top5 = Metric('train_accuracy_top5')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            overall_start = time.time()

            # time the computation time (except step function time)
            train_start = torch.cuda.Event(enable_timing=True)
            train_end = torch.cuda.Event(enable_timing=True)
            train_start.record()

            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                topk_accus = accuracy(output, target_batch, topk=(1, 5))
                train_accuracy.update(topk_accus[0].item())
                train_accuracy_top5.update(topk_accus[1].item())
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss.item())
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

            train_end.record()
            torch.cuda.synchronize()
            computation_time += (train_start.elapsed_time(train_end))

            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            step_start.record()

            # Gradient is applied across all ranks
            optimizer.step()

            step_end.record()
            torch.cuda.synchronize()
            step_time += (step_start.elapsed_time(step_end))

            total_time = time.time() - overall_start

            t.set_postfix({'loss': train_loss.avg.item(),
                           'top1 accuracy': 100. * train_accuracy.avg.item(),
                           'top5 accuracy': 100. * train_accuracy_top5.avg.item(),
                           'throuput': len(data)/total_time})
            throughput_log.append(len(data)/total_time)
            t.update(1)
    # add learning rate, loss, and training accuracies
    accuracy_and_lr.append(optimizer.param_groups[0]['lr'])
    accuracy_and_lr.append(train_loss.avg.item())
    accuracy_and_lr.append(100. * train_accuracy.avg.item())
    accuracy_and_lr.append(100. * train_accuracy_top5.avg.item())

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    val_accuracy_top5 = Metric('val_accuracy_top5')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                topk_accus = accuracy(output, target, topk=(1, 5))
                val_accuracy.update(topk_accus[0].item())
                val_accuracy_top5.update(topk_accus[1].item())
                # val_accuracy.update(accuracy(output, target, topk=(1,5)))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'top1 accuracy': 100. * val_accuracy.avg.item(),
                               'top5 accuracy': 100. * val_accuracy_top5.avg.item()})
                t.update(1)

    # add testing loss and accuracies
    accuracy_and_lr.append(val_loss.avg.item())
    accuracy_and_lr.append(100. * val_accuracy.avg.item())
    accuracy_and_lr.append(100. * val_accuracy_top5.avg.item())

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# BytePS: using `lr = base_lr * bps.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * bps.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
# After the warmup reduce learning rate by 10 on the 20th, 40th, and 60th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / bps.size() * (epoch * (bps.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 20:
        lr_adj = 1.
    elif epoch < 40:
        lr_adj = 1e-1
    elif epoch < 60:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * bps.size() * args.batches_per_pushpull * lr_adj


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.div(output.size(0)))

    # # get the index of the max log-probability
    # pred = output.max(1, keepdim=True)[1]
    # return pred.eq(target.view_as(pred)).float().mean()
    return res


def save_checkpoint(epoch):
    if args.inca or args.new_inca:
        curr_time=datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        try:
            print("storing the error feedback", optimizer._compression.errors)
            with open(compressor_name+'_vgg16_ef'+"_epoch"+str(epoch + 1)+'-'+curr_time+'.pkl', 'wb') as pkl_file:
                pickle.dump(optimizer._compression.errors, pkl_file)
        except:
            print("error feedback of epoch {} not stored".format(epoch + 1))
    if bps.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

def exit_handler():
    print("accuracy, loss, and learning rate:")
    print(accuracy_and_lr)
    print("Throughputs mean, min, max:", sum(throughput_log)/len(throughput_log), min(throughput_log), max(throughput_log))
    if len(throughput_log) < 1000:
        print("Complete throughput records:", throughput_log)
    else:
        print("sample of throughput log:", random.sample(throughput_log, 1000))

# BytePS: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        if args.cuda:
            self.sum = self.sum.cuda()
            self.n = self.n.cuda()

    def update(self, val):
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
            if args.cuda:
                val = val.cuda()
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

atexit.register(exit_handler)

start_time = time.time()
for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    validate(epoch)
    save_checkpoint(epoch)

total_time = time.time() - start_time
print("Total Time: " + str(total_time))
print("Total computation time: " + str(computation_time / 1000))
print("Total time for step function: " + str(step_time / 1000))
print("Throughputs mean, min, max:", sum(throughput_log)/len(throughput_log), min(throughput_log), max(throughput_log))
if len(throughput_log) < 1000:
    print("Complete throughput records:", throughput_log)
else:
    print("sample of throughput log:", random.sample(throughput_log, 1000))