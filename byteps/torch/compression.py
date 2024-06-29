# Copyright 2019 Bytedance Inc. All Rights Reserved.
# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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
"""Gradient compression algorithms."""

import torch
import numpy as np
import warnings
import hadamard_cuda
import time

from scipy.stats import norm


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor, name=None, info=None):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass

################################################################################
################################################################################

class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor, name=None, info=None):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor
    
################################################################################
################################################################################

class DGCCompressor(Compressor):
    def __init__(self, params):
        self.kp = params['kp']
        self.d = params['d']
        self.use_bps_server = params.get('use_bps_server', True)
        self.use_compressor_list = params.get('use_compressor_list', False)
        self.nclients = 1
        self.momentum = params.get('momentum', 0.9)
        self.gradients = {}
        self.residuals = {}
    
    def update_memory(self, tensor, name):
        # update residuals
        if name in self.residuals:
            mmt = self.residuals[name]
            mmt.mul_(self.momentum).add_(tensor)
        else:
            self.residuals[name] = tensor.detach().clone()
        # update accumulated gradient
        if name in self.gradients:
            vec = self.gradients[name]
            vec.add_(mmt)
        else:
            self.gradients[name] = tensor.detach().clone()
            vec = self.gradients[name]
        return vec
    
    def apply_mask(self, name, mask, not_mask):

        self.residuals[name].index_fill_(dim=0, index=mask, value=0)
        self.gradients[name].index_fill_(dim=0, index=mask, value=0)

    """
    This function returns a tensor of indices and values of top coordinates to send, i.e.,
    the i-th element is the index of the top i-th coordinate of the input tensor
    and the (k%*d + i)-th element if the value of the top i-th coordinate.
    """
    def compress(self, tensor, name, info=None):
        all_start = torch.cuda.Event(enable_timing=True)
        all_end = torch.cuda.Event(enable_timing=True)
        all_start.record()

        orig_size = tensor.size()
        tensor = tensor.view(-1)
        d = tensor.numel()
        compressed_num = int(self.kp * d)
        if d < 0:
            res = tensor.detach().clone()
        else:
            res = torch.zeros(size=(d,), device=tensor.device)
            # update residuals and gradients
            tensor = self.update_memory(tensor, name)
            
            # find top values of their indices
            sort, idx = tensor.abs().sort(descending=True)

            res[:compressed_num] = idx[:compressed_num]
            res[compressed_num : 2*compressed_num] = tensor[idx[:compressed_num]]
            mask = idx[:compressed_num]
            not_mask = idx[compressed_num:]
            
            # apply mask
            self.apply_mask(name, mask, not_mask)

        
        all_end.record()
        torch.cuda.synchronize()

        return res, {'size': orig_size, 'compress_overhead': all_start.elapsed_time(all_end), 'name': name}

    def decompress(self, tensor, ctx):
        ## use the code below to do actually decompression
        all_start = torch.cuda.Event(enable_timing=True)
        all_end = torch.cuda.Event(enable_timing=True)
        all_start.record()

        # # Find the number of elements in the original tensor
        d = 1
        for width in ctx['size']:
            d *= width
        compressed_num = int(self.kp * d)

        if d < 0:
            res = tensor.detach().clone()
        else:
            res = torch.zeros(size=(d,), device=tensor.device)

            # index_copy_(dim: _int, index: Tensor, source: Tensor) -> Tensor
            res.index_copy_(0, (tensor[:compressed_num]).long().clamp(max=d-1), tensor[compressed_num:2*compressed_num])

        res /= self.nclients        

        all_end.record()
        torch.cuda.synchronize()
        ctx['decompress_overhead'] = all_start.elapsed_time(all_end)
        return res.view(ctx['size'])

################################################################################
################################################################################

class TopKCompressor(Compressor):
    def __init__(self, params):
        self.kp = params['kp']
        self.ef = params.get('ef', True)
        self.d = params['d']
        self.use_bps_server = params.get('use_bps_server', True)
        self.use_compressor_list = params.get('use_compressor_list', False)
        self.nclients = 1
        
        if self.ef:
            
            self.errors = {}

    """
    This function returns a tensor of tuples (index, value), i.e.,
    the 2*i-th element is the index of the top i-th coordinate of the input tensor
    and the (2*i+1)-th element if the value of the top i-th coordinate.
    """
    def compress(self, tensor, name, info=None):
        all_start = torch.cuda.Event(enable_timing=True)
        all_end = torch.cuda.Event(enable_timing=True)
        all_start.record()

        # print("compress", name)
        # print(tensor)

        orig_size = tensor.size()
        tensor = tensor.view(-1)
        d = tensor.numel()
        compressed_num = int(self.kp * d)
        if d < 0:
            res = tensor.detach().clone()
        else:
            res = torch.zeros(size=(d,), device=tensor.device)

            if self.ef:
                self.errors[name] = tensor + self.errors.get(name, 0)
                sort, idx = self.errors[name].abs().sort(descending=True)
                res[:compressed_num] = idx[:compressed_num]
                res[compressed_num : 2*compressed_num] = self.errors[name][idx[:compressed_num]]

                self.errors[name][idx[:compressed_num]] = 0

            else:
                sort, idx = tensor.abs().sort(descending=True)
                res[:compressed_num] = idx[:compressed_num]
                res[compressed_num : 2*compressed_num] = tensor[idx[:compressed_num]]
        
        all_end.record()
        torch.cuda.synchronize()

        return res, {'size': orig_size, 'compress_overhead': all_start.elapsed_time(all_end), 'name': name}

    def decompress(self, tensor, ctx):
        ## use the code below to do actually decompression
        all_start = torch.cuda.Event(enable_timing=True)
        all_end = torch.cuda.Event(enable_timing=True)
        all_start.record()

        # # Find the number of elements in the original tensor
        d = 1
        for width in ctx['size']:
            d *= width
        compressed_num = int(self.kp * d)

        if d < 0:
            res = tensor.detach().clone()
        else:
            res = torch.zeros(size=(d,), device=tensor.device)

            # index_copy_(dim: _int, index: Tensor, source: Tensor) -> Tensor
            res.index_copy_(0, (tensor[:compressed_num]).long().clamp(max=d-1), tensor[compressed_num:2*compressed_num])

        res /= self.nclients        

        all_end.record()
        torch.cuda.synchronize()
        ctx['decompress_overhead'] = all_start.elapsed_time(all_end)
        # print(res)
        return res.view(ctx['size'])


################################################################################
################################################################################

class TerngradCompressor(Compressor):

    def __init__(self, params=None):

        self.d = params['d']
        self.use_bps_server = params.get('use_bps_server', True)
        self.use_compressor_list = params.get('use_compressor_list', False)
        self.ef = params.get('ef', True)
        self.nclients = 1

        if self.ef:
            
            self.errors = {}
    
    def compress(self, grad, name=None, info=None): 

        orig_size = grad.size()
        grad_max = info["grad_max"]
        if grad_max == 0.0:
            # avoid divide by zero
            grad_max = 1.0
        all_start = torch.cuda.Event(enable_timing=True)
        all_end = torch.cuda.Event(enable_timing=True)
        all_start.record()

        if self.ef:
            self.errors[name] = grad + self.errors.get(name, 0)
            grad = self.errors[name]

            ### extract max and abs => grad_max is s_t
            grad_abs = grad.abs()

            ### randomized rounding
            # each element of b_t indepenently follows the Bernoulli distribution:
            # P(b_tk = 1 | g_t) = |g_tk|/s_t
            # P(b_tk = 0 | g_t) = 1 - |g_tk|/s_t
            p = torch.clamp(grad_abs/grad_max, 0, 1)
            b = torch.bernoulli(p)

            ### pack coordinates
            res = grad.sign()*b+1

            self.errors[name] -= grad_max*(res-1)
        
        else:
    
            ### extract max and abs => grad_max is s_t
            grad_abs = grad.abs()
    
            ### randomized rounding
            # each element of b_t indepenently follows the Bernoulli distribution:
            # P(b_tk = 1 | g_t) = |g_tk|/s_t
            # P(b_tk = 0 | g_t) = 1 - |g_tk|/s_t
            p = torch.clamp(grad_abs/grad_max, 0, 1)
            b = torch.bernoulli(p)

            ### pack coordinates
            res = grad.sign()*b+1
    
        all_end.record()
        torch.cuda.synchronize()

        return res, {'size': orig_size, 'compress_overhead': all_start.elapsed_time(all_end), 'name': name, 'scale': grad_max}

    def decompress(self, tensor, ctx):
        all_start = torch.cuda.Event(enable_timing=True)
        all_end = torch.cuda.Event(enable_timing=True)
        all_start.record()  

        # # Find the number of elements in the original tensor
        d = 1
        for width in ctx['size']:
            d *= width
        
        tensor = tensor.float()

        # Note that we are substracting tensor by nclients since we used 0 for -1, 1 for 0, and 2 for 1
        res = (tensor-self.nclients) * ctx['scale'] / self.nclients

        all_end.record()
        torch.cuda.synchronize()
        ctx['decompress_overhead'] = all_start.elapsed_time(all_end)

        return res.view(ctx['size'])


################################################################################
################################################################################

NUM_ROTATIONS = 20
class Hadamard:
    
    def __init__(self, dim, seed, device):
        
        self.d = dim
        self.device = device
        self.prng = torch.Generator(device=device)
        self.prng.manual_seed(seed)
        self.random_diagonal = 2 * torch.bernoulli(torch.ones(size=(int(2**(np.ceil(np.log2(dim)))),), device=device) / 2, generator=self.prng) - 1
        self.times = 10
            
    def hadamard(self, vec):
        
        d = vec.numel()
       
        return (hadamard_cuda.hadamard_transform(vec, self.times) / np.sqrt(2 ** self.times)).view(-1)

    def rht(self, vec):

        dim = vec.numel()
        
        if not dim & (dim - 1) == 0:
            
            print("WARNING: padding vector in rht")
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = vec
            
            padded_vec = padded_vec * self.random_diagonal[:padded_dim]
            padded_vec = self.hadamard(padded_vec)
            
            return padded_vec
        
        else:   
            
            vec = vec * self.random_diagonal[:dim]
            vec = self.hadamard(vec)
            
            return vec
        
    def irht(self, vec):
        
        vec = self.hadamard(vec)
        vec = vec * self.random_diagonal[:vec.numel()]
        
        return vec


class NewINCACompressor(Compressor):

    def __init__(self, params):

        self.use_bps_server = params.get('use_bps_server', True)
        self.use_compressor_list = params.get('use_compressor_list', False)
        self.device = params.get('device', 'cuda') 

        self.d = params['d']

        if self.d > (8*2**20):
            self.compressor_num = self.d // (8*2**20)
            self.d = (8*2**20)
            self.use_compressor_list = True
        self.seed = params.get('seed', 42)
        
        self.prng = torch.Generator(device=self.device)
        self.prng.manual_seed(self.seed)
                
        self.ef = params.get('ef', True)

        self.quantization_levels = params.get('quantization_levels', 8)
        self.overflow_frequency = params.get('overflow_frequency', 32)
        self.smaxval = params.get('max_val', 16)
        self.table_size = params.get('table_size', 1001)
        
        self.hadamard = Hadamard(self.d, self.seed, self.device)

        if self.ef:
            self.errors = {}
        self.preprocess_holder = {}

        self.fn_prefix = []

        self.fn_prefix.append(params.get('table_dir'))
        self.fn_prefix.append('{}_tablesize_{}_maxval_{}_qlevels_{}_ofreq_'.format(self.table_size,
                                                                      self.smaxval,
                                                                      self.quantization_levels, 
                                                                      self.overflow_frequency))
        
        fn = "/".join(self.fn_prefix)

        ### sender ###########################################################      
        self.sender_prng = torch.Generator(device=self.device)
        self.sender_table_X, self.sender_table_p, self.data = self.sender_table(fn, self.device)
        self.half_table_size = (self.sender_table_X.numel() - 1) // 2
        
         ### receiver #########################################################
        self.receiver_prng = torch.Generator(device=self.device)
        self.recv_table = self.receiver_table(fn, self.device)

        self.nclients = params.get('nclients', 10)
        print("quantization levels", self.quantization_levels)

    def sender_table(self, prefix, device):
    
        sender_table_X = torch.load(prefix + 'sender_table_X.pt').to(device)
        sender_table_p = torch.load(prefix + 'sender_table_p.pt').to(device)
        
        data = eval(open(prefix + 'data.txt').read())
    
        return sender_table_X, sender_table_p, data

    def receiver_table(self, prefix, device):
        
        recv_table = torch.load(prefix +'recv_table.pt').to(device)
        
        return recv_table

    def get_receiver_table(self):
        return self.recv_table

    def take_recv_table(self, vec):
        return torch.take(self.recv_table, vec.long())
            
    def rvec_compress(self, tensor, info=None):
        
        dim = tensor.numel()

        max_coordinate = self.data['T'] # self.data['T'] is t_p in Algorithm 3
        min_coordinate = -max_coordinate
        max_norm = info['max_norm']

        scale = np.sqrt(dim) / max_norm

        # covert coordinates following N(0, max_norm^2/d) to follow N(0, 1), 
        # as the table is for quantizing values following N(0, 1)
        tensor *= scale

        overflow_p = ((tensor > max_coordinate).sum() + (tensor < min_coordinate).sum()) / float(dim)
        if not self.ef and overflow_p > 0:
            warnings.warn('quantization overflow with no error feedback detected: {}% overflow'.format(overflow_p))

        tensor = torch.clamp(tensor, min=min_coordinate, max=max_coordinate)

        tensor /= self.data['delta'] # self.data['delta']=2*t_p/self.table_size=2*self.data['T']/self.table_size

        # Stochastic Quantization
        p = tensor - tensor.floor()
        tensor = tensor.floor() + torch.bernoulli(p, generator=self.sender_prng)

        # Take the sender table
        X = torch.take(self.sender_table_X, (tensor + self.half_table_size).long())
        p_X = torch.take(self.sender_table_p, (tensor + self.half_table_size).long())

        X += torch.bernoulli(p_X).int()

        return X, scale

    def update_record(self, tensor, name):
        if self.ef:
            self.preprocess_holder[name] = tensor.clone() + self.errors.get(name, 0)
        else:
            self.preprocess_holder[name] = tensor.clone()
        return self.preprocess_holder[name].norm(2).item()

    """precompression happen in parallel with max info exchanging"""
    def precompress(self, name):
        """Returns the tensor unmodified."""
        tensor = self.preprocess_holder[name]
        if not tensor.is_cuda:
            print("WARNING: input tensor of THC compress is not GPU tensor")

        # pad tensor here
        dim = tensor.numel()
        
        if not dim & (dim - 1) == 0:
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = tensor
            tensor = padded_vec
            print("WARNING: padding vector in compress", name, dim, padded_dim-dim)

        if self.ef:
            self.errors[name] = tensor.clone()

        self.preprocess_holder[name] = self.hadamard.rht(tensor)
    
    """compression."""
    def compress(self, name, info=None):
        """Returns the tensor unmodified."""
        tensor = self.preprocess_holder[name]
        if not tensor.is_cuda:
            print("WARNING: input tensor of THC compress is not GPU tensor")

        if info['max_norm'] == 0.0:
            tensor.zero_()
            scale = 1.0
        
        else:
        
            if self.ef:
                
                tensor, scale = self.rvec_compress(tensor, info)

                # update the error
                temp2 = torch.take(self.recv_table, tensor.long()).float() - self.smaxval / 2 # self.smaxval is g in Algorithm 3
                self.errors[name] -= self.hadamard.irht(temp2 / scale * self.data['inc']) # self.data['inc'])=2*t_p/g=2*self.data['T']/g
                                            
            else:
                tensor, scale = self.rvec_compress(tensor, info)


        if not tensor.is_cuda:
            print("WARNING: output tensor of INCA compress is not GPU tensor")

        return tensor.float(), {'name': name, 'scale': scale}


    """Uncompress the tensor."""
    def decompress(self, tensor, ctx):
        """Returns the tensor unmodified."""
        if not tensor.is_cuda:
            print("WARNING: input tensor of INCA decompress is not GPU tensor")

        max_norms = ctx['max_norms']

        tensor = tensor.float()
        tensor = tensor.view(-1)
        
        # NOTE: tensor during push-pull has "average=False", so we need to average by nclients here
        tensor =  (tensor / self.nclients) - self.smaxval / 2
        chunk_size = tensor.numel() // ctx['local_size']
        for i in range(ctx['local_size']):
            if max_norms[i] == 0.0:
                tensor[i*chunk_size:(i+1)*chunk_size] = torch.zeros(size=(chunk_size,), device=self.device)
            else:
                scale = np.sqrt(chunk_size) / max_norms[i]
                tensor[i*chunk_size:(i+1)*chunk_size] = \
                    self.hadamard.irht(tensor[i*chunk_size:(i+1)*chunk_size] / scale * self.data['inc'])

        if not tensor.is_cuda:
            print("WARNING: output tensor of INCA decompress is not GPU tensor")
        

        return tensor
    
class INCACompressor(Compressor):
    def __init__(self, params):
        
        self.use_bps_server = params.get('use_bps_server', True)
        self.use_compressor_list = False
        self.device = params.get('device', 'cuda') 
        self.d = params['d']

        if self.d > (8*2**20):
            self.compressor_num = self.d // (8*2**20)
            self.d = (8*2**20)
            self.use_compressor_list = True

        self.seed = params.get('seed', 42)
        
        self.prng = torch.Generator(device=self.device)
        self.prng.manual_seed(self.seed)
        
        self.rotation_prng = torch.Generator(device=self.device)
                
        self.ef = params.get('ef', True)
        self.preprocess_holder = {}
        self.rotation = params.get('rotation', True)

        if self.rotation:
            self.times = params['partial_rotation_times']
        
        self.quantization_levels = params.get('quantization_levels', {})

        if self.ef:
            self.errors = {}
        
        self.norm_normalization = params.get('norm_normalization', False)
        if self.norm_normalization:
            self.per_coordinate_overflow_prob = params.get('per_coordinate_overflow_prob', 0.0001)
        else:
            self.percentile = params['percentile']

        self.nclients = params.get('nclients', 10)

        print("quantization levels", self.quantization_levels)
            
    def hadamard(self, vec):

        if not vec.is_cuda:
            print("WARNING: vec not a GPU tensor in hadamard") 
        
        return (hadamard_cuda.hadamard_transform(vec, self.times) / np.sqrt(2 ** self.times)).view(-1)

    def random_diagonal(self, size, seed):
        
        self.rotation_prng.manual_seed(seed)
        result = 2 * torch.bernoulli(torch.ones(size=(size,), device=self.device) / 2, generator=self.rotation_prng) - 1

        return result

    def randomized_hadamard_transform(self, vec, seed):

        if not vec.is_cuda:
            print("WARNING: input vec of randomized_hadamard_transform is not GPU tensor")

        dim = vec.numel()
        
        if not dim & (dim - 1) == 0:
            
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = vec
            
            temp = self.random_diagonal(padded_vec.numel(), seed)
            padded_vec = padded_vec * temp
            padded_vec = self.hadamard(padded_vec)
            
            if not padded_vec.is_cuda:
                print("WARNING: output vec of randomized_hadamard_transform is not GPU tensor")
            return padded_vec
        
        else:   
            
            temp = self.random_diagonal(vec.numel(), seed)
            vec = vec * temp
            vec = self.hadamard(vec)
            if not vec.is_cuda:
                print("WARNING: output vec of randomized_hadamard_transform is not GPU tensor")
            return vec
        
    def inverse_randomized_hadamard_transform(self, vec, seed, d):
        
        vec = self.hadamard(vec)
        temp = self.random_diagonal(vec.numel(), seed)
        vec = vec * temp
        
        return vec
    
    def norm_stochastic_quantization(self, params):
        vec = params['vec']
        max_norm = params['max_norm']
        quantization_levels = params['quantization_levels']
        
        cloned_vec = vec.clone()

        if not cloned_vec.is_cuda:
            print("WARNING: cloned_vec not a GPU tensor") 
        
        dim = cloned_vec.numel() ### might be padded -> self.d might be wrong
        
        # max_coordinate = norm.isf(self.per_coordinate_overflow_prob , scale=(max_norm/np.sqrt(dim)))
        max_coordinate = norm.isf(self.per_coordinate_overflow_prob , scale=(max_norm/np.sqrt(dim)).cpu())
        min_coordinate = -max_coordinate
                
        delta = (max_coordinate - min_coordinate) / (quantization_levels - 1)
        
        overflow_p = ((cloned_vec > max_coordinate).sum() + (cloned_vec < min_coordinate).sum()) / float(dim)
        if not self.ef and overflow_p > 0:
            warnings.warn('quantization overflow with no error feedback detected: {}% overflow'.format(overflow_p))

        cloned_vec = (cloned_vec - min_coordinate) / delta
        cloned_vec = torch.clamp(cloned_vec, min=0, max=quantization_levels-1)
        cloned_vec = torch.floor(cloned_vec) + torch.bernoulli(cloned_vec-torch.floor(cloned_vec), generator=self.prng)   
        if not cloned_vec.is_cuda:
            print("WARNING: cloned_vec not a GPU tensor after norm quantization") 
                                    
        return cloned_vec, min_coordinate, delta

    def update_record(self, tensor, name):
        if self.ef:
            self.preprocess_holder[name] = tensor.clone() + self.errors.get(name, 0)
        else:
            self.preprocess_holder[name] = tensor.clone()
        return self.preprocess_holder[name].norm(2).item()
    
    """precompression happen in parallel with max info exchanging"""
    def precompress(self, name):
        """Returns the tensor unmodified."""
        tensor = self.preprocess_holder[name]
        if not tensor.is_cuda:
            print("WARNING: input tensor of UHC compress is not GPU tensor")

        dim = tensor.numel()
        
        if not dim & (dim - 1) == 0:
            padded_dim = int(2**(np.ceil(np.log2(dim))))
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = tensor
            tensor = padded_vec
            print("WARNING: padding vector in compress", name, dim, padded_dim-dim)

        if self.ef:
            self.errors[name] = tensor.clone()

        self.preprocess_holder[name] = self.randomized_hadamard_transform(tensor, self.seed)
    
    """compression."""
    def compress(self, name, info=None):
        """Returns the tensor unmodified."""

        tensor = self.preprocess_holder[name]
        if not tensor.is_cuda:
            print("WARNING: input tensor of INCA compress is not GPU tensor")
        orig_size = tensor.size()
        max_norm = info['max_norm']
        if info['max_norm'] == 0.0:
            tensor.zero_()
            min_coordinate = 0.0
            delta = 1.0
        else:
            params = {}
            params['quantization_levels'] = self.quantization_levels.get(name, 16)
            d = tensor.numel()

            sq_func = self.norm_stochastic_quantization
            params['max_norm'] = max_norm

            if self.ef:
                
                if self.rotation:

                    # compress gradient
                    params['vec'] = tensor
                    tensor, min_coordinate, delta = sq_func(params)

                    # update errors
                    temp2 = self.inverse_randomized_hadamard_transform(min_coordinate + tensor * delta, self.seed, d)
                    self.errors[name] -= temp2

                else:

                    # compress gradient
                    params['vec'] = tensor
                    tensor, min_coordinate, delta = sq_func(params)

                    # update errors
                    self.errors[name] -= (min_coordinate + tensor * delta)

            else:
                
                params['vec'] = tensor
                tensor, min_coordinate, delta = sq_func(params)

        if not tensor.is_cuda:
            print("WARNING: output tensor of INCA compress is not GPU tensor")

        return tensor, {'name': name, 'min_coordinate': min_coordinate, 'delta': delta, 'size': orig_size}

    """Upcast the tensor."""
    def decompress(self, tensor, ctx):
        """Returns the tensor unmodified."""
        if not tensor.is_cuda:
            print("WARNING: input tensor of UHC decompress is not GPU tensor")

        min_coordinate = ctx['min_coordinate']
        delta = ctx['delta']
        tensor =  min_coordinate + (tensor / self.nclients) * delta
        tensor = tensor.view(-1)

        # Find the number of elements in the original tensor
        d = 1
        for width in ctx['size']:
            d *= width

        if self.rotation:
            tensor = self.inverse_randomized_hadamard_transform(tensor, self.seed, d)


        if not tensor.is_cuda:
            print("WARNING: output tensor of INCA decompress is not GPU tensor")

        return tensor

class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor, name=None, max_norm=0.0):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        #print("torch fp16 compress")
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        #print("torch fp16 decompress")
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during push_pull."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all gradients using new INCA."""
    newinca = NewINCACompressor

    """Compress all gradients using INCA."""
    inca = INCACompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor

    """Compress all gradients using the dgc compressor."""
    dgc = DGCCompressor

    """Compress all gradients using the topk compressor."""
    topk = TopKCompressor

    """Compress all gradients using the terngrad compressor."""
    terngrad = TerngradCompressor