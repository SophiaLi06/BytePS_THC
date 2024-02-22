# Copyright 2019 ByteDance, Inc. All Rights Reserved.
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

from distutils.version import LooseVersion

# Load all the necessary PyTorch C types.
import torch

# PyTorch must be >= 1.0.0 (including nightly builds)
# This should be guaranteed by setup.py
# TODO: we may not support older pytorch. Raise exception here
from byteps.torch import c_lib
from byteps.common import BytePSBasics as _BytePSBasics
_basics = _BytePSBasics(__file__, 'c_lib')
_NULL = ""


from byteps.torch.compression import Compression

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
suspend = _basics.suspend
resume = _basics.resume
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank


# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}
# keeps track of the norm of the tensor (might be either local or global max_norm)
_name_to_norm = {}
_max_norm = 0
# keeps track of the next tensor to communicate
_name_to_next = {}


def _check_function(function_factory, tensor):
    function = function_factory(tensor)
    if not hasattr(c_lib, function):
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    if not tensor.is_contiguous():
        raise ValueError('Tensor is required to be contiguous.')
    return function


def _push_pull_function_factory(tensor):
    return 'byteps_torch_push_pull_async_' + tensor.type().replace('.', '_')

def _push_pull_group_function_factory(tensor):
    return 'byteps_torch_push_pull_group_sync_' + tensor.type().replace('.', '_')

def _do_push_pull_async(tensor, output, average, name, version=0, priority=0, norm=0.0, compressor_name=""):
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_push_pull_function_factory, tensor)
    # print("push_pull for tensor " + name)
    handle = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, norm, compressor_name.encode())
    _handle_map[handle] = (tensor, output, name)
    return handle

def _do_push_pull_group_sync(tensor, output, average, name, version=0, priority=0):
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL)
    function = _check_function(_push_pull_group_function_factory, tensor)
    # find the norm to send
    norm = tensor.norm(2).item()
    handle, curr_count = getattr(c_lib, function)(tensor, output, average,
                                      name.encode() if name is not None else _NULL,
                                      version, priority, norm)
    _handle_map[handle] = (tensor, output, name)
    return handle, curr_count


def push_pull_async(tensor, average=True, name=None, version=0, priority=0, compressor_name=""):
    """
    A function that performs asynchronous averaging or summation of the input tensor
    over all the BytePS processes. The input tensor is not modified.
    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
    Returns:
        A handle to the push_pull operation that can be used with `poll()` or
        `synchronize()`.
    """
    output = tensor.new(tensor.shape)
    return _do_push_pull_async(tensor, output, average, name, version, priority, _max_norm, compressor_name)


class BytePSPushPull(torch.autograd.Function):
    """An autograd function that performs push_pull on a tensor."""

    @staticmethod
    def forward(ctx, tensor, average, name, version, priority):
        ctx.average = average
        ctx.name = name
        ctx.version = version
        ctx.priority = priority
        handle = push_pull_async(tensor, average, name, version, priority)
        return synchronize(handle)

    @staticmethod
    def backward(ctx, grad_output):
        return push_pull(grad_output,
                         ctx.average, ctx.name, ctx.version, ctx.priority), None, None


def push_pull(tensor, average=True, name=None, version=0, priority=0, compression=Compression.none):
    """
    A function that performs averaging or summation of the input tensor over all the
    BytePS processes. The input tensor is not modified. The reduction operation is keyed
    by the name. The name must be provided. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
        compression: Compression algorithm used during push_pull to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    if name == None:
        raise AssertionError("To manually call push_pull, you must specify a name by name=...")
    tensor_compressed, ctx = compression.compress(tensor)
    summed_tensor_compressed = BytePSPushPull.apply(
        tensor_compressed, average, name, version, priority)
    return compression.decompress(summed_tensor_compressed, ctx)


def push_pull_async_inplace(tensor, average=True, name=None, version=0, priority=0, self_norm=None, compressor_name=""):
    """
    A function that performs asynchronous in-place averaging or summation of the input
    tensor over all the BytePS processes.
    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
    Returns:
        A handle to the push_pull operation that can be used with `poll()` or
        `synchronize()`.
    """
    # if self_norm:
    #     norm = self_norm
    # else:
    #     norm = 0.0
    #     if name in _name_to_next:
    #         next_tensor = _name_to_next[name]
    #         if next_tensor in _name_to_norm:
    #             norm = _name_to_norm[next_tensor]
    #         # print(next_tensor, norm)
    return _do_push_pull_async(tensor, tensor, average, name, version, priority, _max_norm, compressor_name)

def push_pull_group_sync_inplace(tensor, average=True, name=None, version=0, priority=0):
    return _do_push_pull_group_sync(tensor, tensor, average, name, version, priority)

def push_pull_inplace(tensor, average=True, name=None, version=0, priority=0):
    """
    A function that performs in-place averaging or summation of the input tensor over
    all the BytePS processes.
    The reduction operation is keyed by the name. If name is not provided, an incremented
    auto-generated name is used. The tensor type and shape must be the same on all
    BytePS processes for a given name. The reduction will not start until all processes
    are ready to send and receive the tensor.
    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed across all
        processes.
    """
    handle = push_pull_async_inplace(tensor, average, name, version, priority)
    return synchronize(handle)


def poll(handle):
    """
    Polls an push_pull handle to determine whether underlying
    asynchronous operation has completed. After `poll()` returns `True`, `synchronize()`
    will return without blocking.
    Arguments:
        handle: A handle returned by an push_pull asynchronous
                operation.
    Returns:
        A flag indicating whether the operation has completed.
    """
    return c_lib.byteps_torch_poll(handle) != 0


def declare(name):
    c_lib.byteps_torch_declare_tensor(name.encode())
    return 0

def byteps_torch_set_num_grads(num_grads_):
    c_lib.byteps_torch_set_num_grads(num_grads_)
    return 0

def synchronize(handle):
    """
    Synchronizes an asynchronous push_pull operation until
    it's completed. Returns the result of the operation.
    Arguments:
        handle: A handle returned by an push_pull asynchronous
                operation.
    Returns:
        An output tensor of the operation.
    """
    if handle not in _handle_map:
        return
    c_lib.byteps_torch_wait_and_clear(handle)
    _, output, tensor_name = _handle_map.pop(handle)
    # print("done synchronize for " + tensor_name)

    return output

def init_declared_tensor(name, size):
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL)
    c_lib.byteps_torch_init_declared_tensor(name.encode(), size)
    return 0

def send_norm(name, max_norm):
    c_lib.byteps_torch_declare_tensor(name.encode() if name is not None else _NULL)
    return c_lib.byteps_torch_send_norm(name.encode(), max_norm)

def get_norm(name, rank):
    return c_lib.byteps_torch_get_norm(name.encode(), rank)
    # print(norm)

def get_num_worker():
    return c_lib.byteps_torch_get_num_worker()

def get_worker_id():
    return c_lib.byteps_torch_get_worker_id()

def get_local_rank():
    return c_lib.byteps_torch_get_local_rank()

def get_local_size():
    return c_lib.byteps_torch_get_local_size()

def get_partition_bound():
    return c_lib.byteps_torch_get_partition_bound()

def write_norm(name, norm):
    _name_to_norm[name] = norm

def write_next(name, next):
    _name_to_next[name] = next

def clean_records():
    _name_to_next.clear()
    _name_to_norm.clear()

def write_max_norm(norm):
    _max_norm = norm