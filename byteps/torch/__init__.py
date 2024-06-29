# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

from byteps.torch.compression import Compression
from byteps.torch.ops import push_pull_async_inplace as byteps_push_pull
from byteps.torch.ops import push_pull
from byteps.torch.ops import poll, synchronize, declare, init_declared_tensor, get_norm, get_num_worker,\
      get_worker_id, get_local_rank, get_local_size, get_partition_bound
from byteps.torch.ops import init, shutdown, suspend, resume
from byteps.torch.ops import size, local_size, rank, local_rank

import os
import torch
import collections
import io
import cloudpickle

import numpy as np
# import time

class _DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters, compression,
                 backward_passes_per_step=1):
        super(self.__class__, self).__init__(params)
        self._use_compressor_list = False
        self._compression = compression
        self._partition_thres = 8*2**20
        self._total_param = 0
        self.num_workers = get_num_worker()
        self.worker_id = get_worker_id()
        self.local_rank = get_local_rank()
        self.local_size = get_local_size()
        print("BytePS: num_workers: {}, worker_id: {}, local_rank: {}".format(self.num_workers, self.worker_id, self.local_rank))

        if isinstance(self._compression, Compression.newinca):
            print("use INCA compression")
            self._compressor_name = "inca"
            self._use_compressor_list = self._compression.use_compressor_list
            self._compression.nclients = self.num_workers
        elif isinstance(self._compression, Compression.inca):
            print("use old INCA compression")
            self._compressor_name = "oldinca"
            self._use_compressor_list = self._compression.use_compressor_list
            self._compression.nclients = self.num_workers
        elif isinstance(self._compression, Compression.dgc):
            print("use DGC compression")
            self._compressor_name = "dgc"
            self._compression.nclients = self.num_workers
        elif isinstance(self._compression, Compression.topk):
            print("use TopK compression")
            self._compressor_name = "topk"
            self._compression.nclients = self.num_workers
        elif isinstance(self._compression, Compression.terngrad):
            print("use TernGrad compression")
            self._compressor_name = "terngrad"
            self._compression.nclients = self.num_workers
        else:
            print("use no compression")
            self._compression = Compression.none()
            self._compressor_name = "none"
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        self._total_param = sum(p[1].numel() for p in named_parameters)
        print("total number of elements in named_parameters: ", self._total_param)
        
        self.uncompressed_batch_tensors = []
        self.slice_size = []
        self.compressed_batch_tensors = []
        self.max_info_tensor = []
        bound = get_partition_bound() // 4
        print("Partition bound: ", bound)
        if self._total_param <= bound:
            if isinstance(self._compression, Compression.newinca) or isinstance(self._compression, Compression.inca):
                length_with_pad = int(2**(np.ceil(np.log2(self._total_param))))
                self._total_param = length_with_pad
            if isinstance(self._compression, Compression.none):
                self.uncompressed_batch_tensors.append(torch.zeros(self._total_param).cuda())
                init_declared_tensor("Gradient.Uncompress_batched_0", self._total_param)
            self.slice_size.append(self._total_param)
            if isinstance(self._compression, Compression.newinca) or isinstance(self._compression, Compression.inca)\
                  or isinstance(self._compression, Compression.terngrad):
                self.compressed_batch_tensors.append((torch.zeros(self._total_param).cuda()).to(torch.uint8))
                self.max_info_tensor.append(torch.zeros(8*8+8).cuda())
                init_declared_tensor("Gradient.Compress_batched_0", self._total_param)
            elif isinstance(self._compression, Compression.topk) or isinstance(self._compression, Compression.dgc):
                self.compressed_batch_tensors.append(torch.zeros(2*int(self._total_param * self._compression.kp)).cuda())
                init_declared_tensor("Gradient.Sparsek_batched_0", self._total_param)
        else:
            
            for i in range(0, self._total_param // bound):
                if isinstance(self._compression, Compression.none):
                    self.uncompressed_batch_tensors.append(torch.zeros(bound).cuda())
                    init_declared_tensor("Gradient.Uncompress_batched_{}".format(i), bound)
                self.slice_size.append(bound)
                if isinstance(self._compression, Compression.newinca) or isinstance(self._compression, Compression.inca)\
                      or isinstance(self._compression, Compression.terngrad):
                    self.compressed_batch_tensors.append((torch.zeros(bound).cuda()).to(torch.uint8))
                    self.max_info_tensor.append(torch.zeros(8*8+8).cuda())
                    init_declared_tensor("Gradient.Compress_batched_{}".format(i), bound)
                elif isinstance(self._compression, Compression.topk) or isinstance(self._compression, Compression.dgc):
                    self.compressed_batch_tensors.append(torch.zeros(2*int(bound * self._compression.kp)).cuda())
                    init_declared_tensor("Gradient.Sparsek_batched_{}".format(i), bound)
            # make sure that the last tensor's length is still divisible by 8
            length_with_pad = int(2**(np.ceil(np.log2(self._total_param % bound))))
            self._total_param += length_with_pad - self._total_param % bound
            if isinstance(self._compression, Compression.none):
                self.uncompressed_batch_tensors.append(torch.zeros(length_with_pad).cuda())
                init_declared_tensor("Gradient.Uncompress_batched_{}".format(len(self.slice_size)), length_with_pad)
            self.slice_size.append(length_with_pad)
            if isinstance(self._compression, Compression.newinca) or isinstance(self._compression, Compression.inca)\
                  or isinstance(self._compression, Compression.terngrad):
                self.compressed_batch_tensors.append((torch.zeros(length_with_pad).cuda()).to(torch.uint8))
                self.max_info_tensor.append(torch.zeros(8*8+8).cuda())
                init_declared_tensor("Gradient.Compress_batched_{}".format(len(self.slice_size)-1), length_with_pad)
            elif isinstance(self._compression, Compression.topk) or isinstance(self._compression, Compression.dgc):
                self.compressed_batch_tensors.append(torch.zeros(2*int(length_with_pad * self._compression.kp)).cuda())
                init_declared_tensor("Gradient.Sparsek_batched_{}".format(len(self.slice_size)-1), length_with_pad)

        if isinstance(self._compression, Compression.newinca) or isinstance(self._compression, Compression.inca)\
              or isinstance(self._compression, Compression.terngrad):
            self.max_info_tensor = torch.cat(self.max_info_tensor)

        self._enable_async = (int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
        if self._enable_async:
            assert int(os.getenv('DMLC_NUM_WORKER')) > 1, \
                "Async is only valid for distributed training"
            print('BytePS: enable asynchronous training')

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _DistributedOptimizer.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        if len(named_parameters) > 0:
            if isinstance(named_parameters[0][1], torch.Tensor):
                if any([not isinstance(p, torch.Tensor) for name, p in named_parameters]):
                    raise ValueError('named_parameters should consistently be a sequence of '
                                     'tuples (name, torch.Tensor)')
                self._is_tensor_instance = True
                # there is an issue when using torch.Tensor as key, so use its hash instead
                # https://github.com/pytorch/pytorch/issues/7733
                self._parameter_names = {v.__hash__(): k for k, v
                                         in sorted(named_parameters)}
                self._tensor_list = [tensor for name, tensor in named_parameters]
            else:
                self._is_tensor_instance = False
                self._parameter_names = {v: k for k, v
                                         in sorted(named_parameters)}
        else:
            self._is_tensor_instance = False
            self._parameter_names = {v: 'push_pull.noname.%s' % i
                                     for param_group in self.param_groups
                                     for i, v in enumerate(param_group['params'])}
        self.backward_passes_per_step = backward_passes_per_step
        self._push_pull_delay = {v: self.backward_passes_per_step
                                 for _, v in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []
        self._requires_update = set()
        self._should_sync = True
        self._register_hooks()

        self._compression_info = {}
        self.compress_overhead = 0.0
        self.decompress_overhead = 0.0
        self.compress_diag_time = 0.0
        self.compress_hadamard_time = 0.0
        self.decompress_diag_time = 0.0
        self.decompress_hadamard_time = 0.0

        # declare tensors
        for name in sorted(self._parameter_names.values()):
            declare("Gradient."+name)
        # We use two loops for load-balancing
        for name in sorted(self._parameter_names.values()):
            declare("Parameter."+name)

    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups

    def set_backward_passes_per_step(self, passes):
        self.backward_passes_per_step = passes
        for p in self._push_pull_delay:
            self._push_pull_delay[p] = self.backward_passes_per_step

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p.grad = p.data.new(p.size()).zero_()
                    self._requires_update.add(p)
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)
    
    def _aftercompress_push_pull_grad_async(self, tensor, index):
        # NOTE: we don't average here because data are of torch.uint8 and would got rounded down
        handle = byteps_push_pull(tensor.to(torch.uint8), average=False,\
                        name="Gradient.Compress_batched_{}".format(index), compressor_name=self._compressor_name)
        return handle

    def _push_pull_grad_async(self, p):
        if self._is_tensor_instance:
            name = self._parameter_names.get(p.__hash__())
        else:
            name = self._parameter_names.get(p)
        if self._enable_async:
            # the real handle will be created in step()
            handle, ctx = None, None
        else:
            tensor = p.grad
            tensor_compressed, ctx = self._compression.compress(tensor)
            handle = byteps_push_pull(tensor_compressed, average=True, name="Gradient."+name)
        return handle, ctx
    
    def _push_pull_max_info(self, max_vals):
        self.max_info_tensor.zero_()
        for i in range(len(max_vals)):
            self.max_info_tensor[i*(8*8+8) + self.local_rank * 8 + self.worker_id] = max_vals[i]
        # we want to find the maximum value, so no averaging
        handle = byteps_push_pull(self.max_info_tensor, average=False, \
                                  name="Max_info_batched")
        return handle
    
    def _push_pull_max_norm(self, max_norm, tensor_idx):
        self.max_info_tensor[tensor_idx].zero_()
        self.max_info_tensor[tensor_idx][self.local_rank * 8 + self.worker_id] = max_norm
        # we want to find the maximum value, so no averaging
        handle = byteps_push_pull(self.max_info_tensor[tensor_idx], average=False, \
                                  name="Max_info_{}".format(tensor_idx))
        return handle
    
    def _push_pull_max_info_terngrad(self, max_info):
        self.max_info_tensor.zero_()
        self.max_info_tensor[self.worker_id] = max_info
        # we want to find the maximum value, so no averaging
        handle = byteps_push_pull(self.max_info_tensor, average=False, \
                                  name="Max_info_terngrad")
        return handle
    
    def _sparsek_push_pull_grad_async(self, tensors):
        handles = []
        for i in range(len(self.compressed_batch_tensors)):
            self.compressed_batch_tensors[i].zero_()
            self.compressed_batch_tensors[i] = tensors[i][:len(self.compressed_batch_tensors[i])]
            # NOTE: we don't average here
            handles.append(byteps_push_pull(self.compressed_batch_tensors[i], average=False,\
                            name="Gradient.Sparsek_batched_{}".format(i), compressor_name=self._compressor_name))
        return handles

    def _terngrad_push_pull_grad_async(self, tensors):
        handles = []
        for i in range(len(self.compressed_batch_tensors)):
            self.compressed_batch_tensors[i].zero_()
            self.compressed_batch_tensors[i] = (tensors[i].to(torch.uint8))[:len(self.compressed_batch_tensors[i])]
            # NOTE: we don't average here
            handles.append(byteps_push_pull(self.compressed_batch_tensors[i], average=False,\
                            name="Gradient.Compress_batched_{}".format(i), compressor_name=self._compressor_name))
        return handles
    
    def _uncompress_batch_push_pull_grad_async(self, tensor, average=True):
        handles = []
        offset = 0
        for i in range(len(self.uncompressed_batch_tensors)):
            slice_size = len(self.uncompressed_batch_tensors[i])
            self.uncompressed_batch_tensors[i] = tensor[offset:offset + slice_size].detach().clone()
            offset += slice_size
            handles.append(byteps_push_pull(self.uncompressed_batch_tensors[i], average=average,\
                                             name="Gradient.Uncompress_batched_{}".format(i), \
                                             compressor_name=self._compressor_name))
        return handles

    def _make_hook(self, p):
        def hook(*ignore):
            if p in self._handles and self._handles[p][0] is not None:
                if self._push_pull_delay[p] <= 0:
                    raise AssertionError(
                        "Gradients were computed more than "
                        "backward_passes_per_step times before call "
                        "to step(). Increase backward_passes_per_step to "
                        "accumulate gradients locally.")
            assert not p.grad.requires_grad
            assert self._push_pull_delay[p] > 0
            handle, ctx = None, None
            self._push_pull_delay[p] -= 1
            self._handles[p] = (handle, ctx)
        return hook

    def synchronize(self):

        missing_p = self._requires_update - set(self._handles.keys())
        for p in missing_p:
            handle, ctx = None, None
            self._handles[p] = (handle, ctx)
        
        grads = []
        for p, value in self._handles.items():
            grads.append(p.grad.view(-1))
        grads = torch.cat(grads)
        # pad grads to be divisible by 8
        if len(grads) < self._total_param:
            grads = torch.cat([grads, torch.zeros(self._total_param-len(grads)).cuda()])

        if isinstance(self._compression, Compression.newinca) \
            or isinstance(self._compression, Compression.inca):

            norm_vals = []
            offset = 0
            
            for i in range(len(self.slice_size)):
                slice_size = self.slice_size[i]
                norm_vals.append(self._compression.update_record(grads[offset:offset + slice_size], 
                                                                 "batch_grads_{i}".format(i=i)))
                offset += slice_size
                
            max_info_handle = self._push_pull_max_info(norm_vals)
            # do tensor rotation in parallel
            offset = 0
            if not grads.is_cuda:
                print("WARNING: grad to compress is not on CUDA")
            for i in range(len(self.slice_size)):
                slice_size = self.slice_size[i]
                self._compression.precompress("batch_grads_{i}".format(i=i))
                offset += slice_size
            max_info_output = synchronize(max_info_handle)

            max_norms = []
            for i in range(len(self.slice_size)):
                start_idx = (8*8+8) * i
                max_norms.append(\
                    [max(max_info_output[start_idx + 8 * x : start_idx + 8 * (x + 1)]) for x in range(self.local_size)])

            contexts = []
            compress_batch_handles = []
            offset = 0
            if not grads.is_cuda:
                print("WARNING: grad to compress is not on CUDA")
            for i in range(len(self.slice_size)):
                slice_size = self.slice_size[i]
                compressed_slice, ctx = self._compression.compress(\
                    "batch_grads_{i}".format(i=i), \
                    {'max_norm':max_norms[i][self.local_rank], "dim":slice_size})
                compress_batch_handles.append(self._aftercompress_push_pull_grad_async(compressed_slice, i))
                contexts.append(ctx)
                offset += slice_size

            batch_decompressed = []
            for i in range(len(compress_batch_handles)):
                handle = compress_batch_handles[i]
                output = synchronize(handle)
                if not output.is_cuda:
                    print("WARNING: pulled tensor is not on CUDA")
                contexts[i]['max_norms'] = max_norms[i]
                contexts[i]['local_size'] = self.local_size
                batch_decompressed.append(self._compression.decompress(output, contexts[i]))

            batch_decompressed = torch.cat(batch_decompressed)

            offset = 0
            for p, _ in self._handles.items():
                slice_size = len(p.grad.view(-1))
                grad_slice = batch_decompressed[offset:offset + slice_size]
                offset += slice_size
                p.grad.set_(grad_slice.resize_(p.grad.shape))
                self._push_pull_delay[p] = self.backward_passes_per_step

        elif isinstance(self._compression, Compression.dgc) or isinstance(self._compression, Compression.topk)\
            or isinstance(self._compression, Compression.terngrad):
            offset = 0
            max_vals = []

            if isinstance(self._compression, Compression.terngrad):
                for i in range(len(self.slice_size)):
                    slice_size = self.slice_size[i]
                    chunk_max = grads[offset:offset + slice_size].abs().max().item()
                    max_vals.append(chunk_max)
                    offset += slice_size
            
                # find the global scale of each chunk
                max_infos = []
                max_info_output = synchronize(self._push_pull_max_info(max_vals))
                for i in range(len(self.slice_size)):
                    max_infos.append(max_info_output[i*(8*8+8):(i+1)*(8*8+8)].max().item())

            batch_compressed = []
            contexts = []
            offset = 0
            if not grads.is_cuda:
                print("WARNING: grad to compress is not on CUDA")
            for i in range(len(self.slice_size)):
                slice_size = self.slice_size[i]
                if isinstance(self._compression, Compression.terngrad):
                    compressed_chunk, ctx = self._compression.compress(\
                        grads[offset:offset + slice_size],"batch_grads_{i}".format(i=i), \
                        {'grad_max':max_infos[i]})
                else:
                    compressed_chunk, ctx = self._compression.compress(\
                        grads[offset:offset + slice_size],"batch_grads_{i}".format(i=i))
                batch_compressed.append(compressed_chunk)
                contexts.append(ctx)
                offset += slice_size

            batched_output = []
            if isinstance(self._compression, Compression.dgc) or isinstance(self._compression, Compression.topk):
                compress_batch_handles =self._sparsek_push_pull_grad_async(batch_compressed)
                for handle in compress_batch_handles:
                    batched_output.append(synchronize(handle).view(-1))
            else:
                compress_batch_handles =self._terngrad_push_pull_grad_async(batch_compressed)
                for handle in compress_batch_handles:
                    batched_output.append(synchronize(handle).view(-1))

            if not self._enable_async:
                batch_decompressed = []
                for i in range(len(batched_output)):
                    output = batched_output[i]
                    if not output.is_cuda:
                        print("WARNING: pulled tensor is not on CUDA")
                    batch_decompressed.append(self._compression.decompress(batched_output[i], contexts[i]))
                    
                    self.compress_overhead += contexts[i]['compress_overhead']
                    self.decompress_overhead += contexts[i]['decompress_overhead']
                    
                batch_decompressed = torch.cat(batch_decompressed)

                offset = 0
                for p, _ in self._handles.items():
                    slice_size = len(p.grad.view(-1))
                    grad_slice = batch_decompressed[offset:offset + slice_size]
                    offset += slice_size
                    p.grad.set_(grad_slice.resize_(p.grad.shape))
                    # reset the pushpull delay
                    self._push_pull_delay[p] = self.backward_passes_per_step

        else:
            
            uncompress_batch_handles = self._uncompress_batch_push_pull_grad_async(grads)
            batched_output = []
            for handle in uncompress_batch_handles:
                batched_output.append(synchronize(handle))
            batched_output = torch.cat(batched_output)
            offset = 0
            for p, _ in self._handles.items():
                slice_size = len(p.grad.view(-1))
                grad_slice = batched_output[offset:offset + slice_size]
                offset += slice_size
                p.grad.set_(grad_slice.resize_(p.grad.shape))
                self._push_pull_delay[p] = self.backward_passes_per_step         

        self._handles.clear()


    @contextmanager
    def skip_synchronize(self):
        if self._enable_async:
            raise AssertionError("skip_synchronize cannot be used in async training")
        self._should_sync = False
        try:
            yield
        finally:
            self._should_sync = True

    def step(self, closure=None):
        if self._enable_async:
            old_weight_map = {}
            # store the weights before update
            for p, _ in self._handles.items():
                old_weight_map[p] = p.data.clone().detach()
            # update
            loss = super(self.__class__, self).step(closure)

            for p, (h, _) in self._handles.items():
                # get the diff for each weight (in-place)
                p.data.sub_(old_weight_map.get(p))
                if h is None:
                    # create the handler now
                    if self._is_tensor_instance:
                        name = self._parameter_names.get(p.__hash__())
                    else:
                        name = self._parameter_names.get(p)
                    handle = byteps_push_pull(p, average=False, name="AsyncParam."+name)
                    _, ctx = self._compression.compress(p)
                    self._handles[p] = (handle, ctx)

            self.synchronize()
            return loss
        else:
            # skip sync if calling skip_synchronize
            if self._should_sync:
                self.synchronize()
            return super(self.__class__, self).step(closure)


def DistributedOptimizer(optimizer, named_parameters=None,
                         compression=Compression.none,
                         backward_passes_per_step=1):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an push_pull to
    average gradient values before applying gradients to model weights.
    push_pull operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all push_pull operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces push_pull operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          push_pull operations. Typically just `model.named_parameters()`.
        compression: Compression algorithm used during push_pull to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        backward_passes_per_step: Number of expected backward passes to perform
                                  before calling step()/synchronize(). This
                                  allows accumulating gradients over multiple
                                  mini-batches before executing averaging and
                                  applying them.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an push_pull implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,
               compression, backward_passes_per_step)


def broadcast_parameters(params, root_rank, prefix="Parameter."):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.
    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run synchronous broadcasts.
    for name, p in params:
        # Broadcast is implemented as push + pull in BytePS
        # To make it a real broadcast, we set the non-root tensors all 0.
        if rank() != root_rank:
            p.fill_(0)
        # Remember to disable averaging because we are doing broadcast
        if name:
            handle = byteps_push_pull(p, average=False, name=prefix+name)
        else:
            handle = byteps_push_pull(p, average=False)
        synchronize(handle)


def broadcast_optimizer_state(optimizer, root_rank, prefix="Parameter."):
    """
    Broadcasts an optimizer state from root rank to all other processes.
    Arguments:
        optimizer: An optimizer.
        root_rank: The rank of the process from which the optimizer will be
                   broadcasted to all other processes.
    """
    if isinstance(optimizer, torch.optim.LBFGS):
        # TODO(travis): L-BFGS cannot be easily supported without serializing
        # the entire state_dict, as its structure is deeply nested and contains
        # None type parameter values
        raise ValueError('cannot broadcast torch.optim.LBFGS state')

    state_dict = optimizer.state_dict()

    # Newly created optimizers will not have their state initialized, so
    # do that initialization here
    if len(state_dict['state']) == 0:
        for group in optimizer.param_groups:
            for p in group['params']:
                p.grad = p.data.new(p.size()).zero_()
        # This function accepts a torch.optim.Optimizer or a DistributedOptimizer
        # wrapped around a torch optimizer. Calling step() with a DistributedOptimizer
        # forces push_pull on all model parameters, which will result in deadlock
        # unless every rank calls step(). Therefore, to finish state initialization
        # only call optimizer.step() with a torch.optim.Optimizer.
        if optimizer.__module__ == DistributedOptimizer.__module__:
            super(optimizer.__class__, optimizer).step()
        else:
            optimizer.step()
        state_dict = optimizer.state_dict()

    # If the state_dict is still empty after initialization, then
    # the optimizer is stateless, and there is nothing to broadcast.
    # Furthermore, attempting to access the state dict would result in
    # an error.
    if len(state_dict['state']) == 0:
        return

    params = []
    scalars = {}
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    # Some optimizer parameters may be represented as scalars instead of
    # tensors.  In such cases, we place the scalars into a single dict,
    # then pickle and broadcast with broadcast_object (under the assumption
    # that there are not many scalars, and so the overhead of pickling will
    # be relatively low). Because broadcast_object is performed out-of-place,
    # we then use a callback to assign the new value to the correct element
    # of the optimizer state.
    def _create_state_callback(pid, name):
        def _assign_state(v):
            state_dict['state'][pid][name] = v
        return _assign_state

    def _create_option_callback(index, option_key):
        def _assign_option(v):
            optimizer.param_groups[index][option_key] = v
        return _assign_option

    # Param groups are an ordered list, normally there is only one per model,
    # but users can add additional param groups for example to train
    # previously frozen layers
    for index, group in enumerate(state_dict['param_groups']):
        # Broadcast options like learning rate
        for option_key, option_value in group.items():
            if option_key == 'params':
                continue

            # Options like the learning rate are scalar, and need to be broadcast separately
            key = '%s.%d' % (option_key, index)
            dtypes = _get_types(option_value)
            option_tensor = torch.Tensor([option_value]).cuda()
            scalars[key] = option_value
            callbacks[key] = _create_option_callback(index, option_key)

        # The params list here is ordered by the layers in the model
        for pid in group['params']:
            if pid not in state_dict['state']:
                # The param has not set requires_grad, so skip broadcast
                continue

            param_state = state_dict['state'][pid]
            for name, p in param_state.items():
                # Some parameter names may appear more than once, in which
                # case we ensure they have a unique identifier defined by
                # their order
                occurrences[name] += 1
                key = '%s.%d' % (str(name), occurrences[name])

                if torch.is_tensor(p):
                    # Tensor -> use broadcast_parameters
                    params.append((key, p))
                else:
                    # Scalar -> use broadcast_object
                    scalars[key] = p
                    callbacks[key] = _create_state_callback(pid, name)

    # Synchronized broadcast of all parameters
    broadcast_parameters(params, root_rank, prefix)

    # Broadcast and cleanup for non-tensor parameters
    scalars = broadcast_object(scalars, root_rank)
    for key, p in scalars.items():
        callbacks[key](p)

def broadcast_object(obj, root_rank=0, name=None):
    """
    Serializes and broadcasts an object from root rank to all other processes.
    Typical usage is to broadcast the `optimizer.state_dict()`, for example:
    .. code-block:: python
        state_dict = broadcast_object(optimizer.state_dict(), 0)
        if bps.rank() > 0:
            optimizer.load_state_dict(state_dict)
    Arguments:
        obj: An object capable of being serialized without losing any context.
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
        name: Optional name to use during broadcast, will default to the class
              type.
    Returns:
        The object that was broadcast from the `root_rank`.
    """
    if name is None:
        name = type(obj).__name__

    if rank() == root_rank:
        b = io.BytesIO()
        cloudpickle.dump(obj, b)
        t = torch.ByteTensor(bytearray(b.getvalue()))
        sz = torch.IntTensor([t.shape[0]])
        broadcast_parameters([(name + '.sz', sz)], root_rank, prefix="Size.")
    else:
        sz = torch.IntTensor([0])
        broadcast_parameters([(name + '.sz', sz)], root_rank, prefix="Size.")
        t = torch.ByteTensor(sz.tolist()[0])

    broadcast_parameters([(name + '.t', t)], root_rank, prefix="Parameter.")

    if rank() != root_rank:
        buf = io.BytesIO(t.numpy().tobytes())
        obj = cloudpickle.load(buf)

    return obj
