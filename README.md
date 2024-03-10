# THC

THC (Tensor Homomorphic Compression) is a novel bi-directional compression framework that enables the direct aggregation of compressed values and thus eliminating the computational overheads of compress/decompress data on parameter servers.

## Recommend Versions
+ Python >= 3.8
+ CUDA 11.3
+ PyTorch 1.10.1
+ NCCL 2.11.4

## Installation
```
git clone --recursive https://github.com/SophiaLi06/BytePS_THC.git
cd BytePS_THC
python3 setup.py install
```
As mentioned in the [BytePS repository](https://github.com/bytedance/byteps), please specify your NCCL path with `export BYTEPS_NCCL_HOME=/path/to/nccl`.

## Publications

+ [NSDI'24] "THC: Accelerating Distributed Deep Learning Using Tensor Homomorphic Compression". Minghao Li, Ran Ben Basat, Shay Vargaftik, ChonLam Lao, Kevin Xu, Michael Mitzenmacher, Minlan Yu 
