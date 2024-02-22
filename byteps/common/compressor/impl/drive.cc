#include <cstring>
#include <stdio.h>
#include <iostream>

#include "../compressor_registry.h"
#include "drive.h"

namespace byteps {
namespace common {
namespace compressor {
namespace{
CompressorRegistry::Register reg(
    "drive_compressor",
    [](const kwargs_t& kwargs, size_t size,
       DataType dtype) -> std::unique_ptr<Compressor>{

      auto seed = HyperParamFinder<unsigned>(kwargs, "seed", true,
                                            [](unsigned x) {return x > 0;});
    
      return std::unique_ptr<Compressor>(
          new DriveCompressor(size, dtype, seed));
    });
}

/*
 * Except for error-feedback and momentum, the underlying data of input
 * should never be changed. this is because input is still used in error
 * feedback if enabled.
 */

/* In-Place 1D Hadamard Rotate*/
template <typename index_t, typename scalar_t>
void DriveCompressor::HadamardRotate(index_t* dst, const scalar_t* src,
                                       size_t len) {

  auto start = std::chrono::high_resolution_clock::now();
  
  assert(len & (len-1) == 0);
  size_t h = 2;
  size_t hf;
  //TODO: can this process be paralleled in some way?
  while (h <= len){
    hf = h / 2;
    // view the gradient as a (len // h * h) tensor
    for (size_t i = 0; i < len / h; i++){
      for (size_t j = 0; j < hf; j++) {
        // update front half of each "row"
        dst[i * h + j] = dst[i * h + j] + dst[i * h + hf + j];
        // update back half of each "row"
        dst[i * h + hf + j] = dst[i * h + j] - 2 * dst[i * h + hf + j];
      }
    }
    h *= 2;
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(this->_rotate_mtx);
  this->_rotate_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

template <typename index_t, typename scalar_t>
tensor_t DriveCompressor::CompressImpl(index_t* dst, const scalar_t* src,
                                       size_t len) {
  auto start = std::chrono::high_resolution_clock::now();

  static_assert(sizeof(index_t) == sizeof(scalar_t),
                "index_t should be the same size as scalar_t");

  // PACKING_SIZE values will be compressed into one chunk
  // (Each scalar value is represented by one bit, so 8 values in one byte
  // and sizeof(scalar_t) * 8 in one scalar_t)
  constexpr size_t PACKING_SIZE = sizeof(scalar_t) * 8;
  size_t padding_len = (PACKING_SIZE - (len % PACKING_SIZE)) % PACKING_SIZE;
  // The total number of chunks
  const size_t chunk_num = (len + padding_len) / PACKING_SIZE;
  
  // In-Place 1D Hadamard Rotate
  // TODO: may need to modify len to make it into a power of 2?
  scalar_t temp[len];

  // TODO: originally is sqaure root of len, but I need to keep this the same
  // for both compress and decompress, so add a padding_len? Confirm this!!!
  float sqrt_d = std::sqrt(len);

  if (_seed != 0){
    // if random number generator is not none
    for (size_t i = 0; i < len; i++){
      //temp[i] = src[i] * (2 * _rng.Bernoulli(0.5) - 1);
      if (_rng.Bernoulli(0.5)) {temp[i] = src[i]; this->_bernoulli++;}
      else {temp[i] = -src[i];}
      temp[i] /= sqrt_d;
    }
  }
  else{
    for (size_t i = 0; i < len; i++){
      // if (_rng.Bernoulli(0.5)) {temp[i] = src[i];}
      // else {temp[i] = -src[i];}
      // temp[i] /= sqrt_d;
      temp[i] = src[i]/sqrt_d;
    }
  }

  HadamardRotate(temp, temp, len);

  // Compute the scale
  double norm1 = 0.0f, norm2 = 0.0f;
  for (size_t i = 0; i < len; i++){
    norm1 += std::abs(temp[i]);
    norm2 += (temp[i] * temp[i]);
  }
  //note norm2 is actually the square of the L2 norm
  float scale = norm2 / norm1;

#pragma omp parallel for simd
  for (size_t i = 0; i < chunk_num; i++){
    size_t start_index = i * PACKING_SIZE;
    index_t x = temp[start_index] < 0;

    for (size_t j = 1; j < PACKING_SIZE; j++){
      x <<= 1;
      // take the sign
      // ('0' for positve, '1' for negative)
      x |= (temp[start_index + j] < 0);
    }
    dst[i] = x;
  }

  // append the scale to the end of the tensor
  float* scale_ptr = reinterpret_cast<float*>(&dst[chunk_num]);
  *scale_ptr = scale;

  auto end = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(this->_compress_mtx);
  this->_compress_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  this->_compress_call++;

  return {dst, chunk_num * sizeof(index_t) + sizeof(float)};
}

tensor_t DriveCompressor::Compress(tensor_t grad) {
  COMPRESS_IMPL_SWITCH(grad.dtype, CompressImpl, _buf.get(), grad.data,
                       grad.size);
}

template <typename scalar_t, typename index_t>
tensor_t DriveCompressor::DecompressImpl(scalar_t* dst, const index_t* src,
                                        size_t compressed_size){
  auto start = std::chrono::high_resolution_clock::now();

  static_assert(sizeof(scalar_t) == sizeof(index_t),
                "scalar_t should be the same size as index_t");
  constexpr size_t PACKING_SIZE = sizeof(scalar_t) * 8;
  const size_t chunk_num = (compressed_size - sizeof(float)) / sizeof(index_t);

  auto* scale_ptr = reinterpret_cast<const float*>(src + chunk_num);
  float scale = *scale_ptr;

  index_t* ptr = const_cast<index_t*>(src);
  if ((void*)dst == (void*)src) {
    ptr = reinterpret_cast<index_t*>(_buf.get());
    std::memcpy(ptr, src, compressed_size);
  }

#pragma omp parallel for simd
  for (int i = chunk_num - 1; i >= 0; i--){
    index_t x = ptr[i];
    for (int j = PACKING_SIZE - 1; j >= 0; j--){
      // restore the sign
      // (1 for positive, -1 for negative)
      // TODO: not casting to float should be fine? as it will then be
      // divided by the float "sqrt_d" in HadamardRotate?
      int sign = 1 - ((x & 0x01) << 1);
      dst[i * PACKING_SIZE + j] = sign;
      x >>= 1;
    }
  }

  // in-place Hadamard Transform (inverse)
  HadamardRotate(dst, dst, chunk_num * PACKING_SIZE);

  float sqrt_d = std::sqrt(chunk_num * PACKING_SIZE);

  if (_seed != 0){
    // if random number generator is not none
    for (size_t i = 0; i < chunk_num * PACKING_SIZE; i++){
      if (_rng.Bernoulli(0.5)) {dst[i] = dst[i]/sqrt_d;}
      else {dst[i] = -dst[i]/sqrt_d;}
      //dst[i] = dst[i] * (2 * _rng.Bernoulli(0.5) - 1);
    }
  }
  else{
    for (size_t i = 0; i < chunk_num * PACKING_SIZE; i++){
      dst[i] = dst[i]/sqrt_d;
    }
  }

  // TODO: remove the for loop below !!!!!!!!
  // for (size_t i = 0; i < chunk_num * PACKING_SIZE; i++){
  //   if (_rng.Bernoulli(0.5)) {dst[i] = dst[i]/sqrt_d;}
  //   else {dst[i] = -dst[i]/sqrt_d;}
  // }

  // scale and return
  for (size_t i = 0; i < chunk_num * PACKING_SIZE; i++){
    dst[i] *= scale;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::lock_guard<std::mutex> lock(this->_decompress_mtx);
  this->_decompress_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
  this->_decompress_call++;

  return {dst, _size};
}

tensor_t DriveCompressor::Decompress(tensor_t compressed) {
#ifdef BYTEPS_BUILDING_SERVER
  auto dst = _buf.get();
#else
  auto dst = compressed.data;
#endif

  this->_decompress_size = compressed.size;
  DECOMPRESS_IMPL_SWITCH(_dtype, DecompressImpl, dst, compressed.data,
                         compressed.size);
}

}  // namespace compressor
}  // namespace common
}  // namespace byteps
