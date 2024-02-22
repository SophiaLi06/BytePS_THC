#ifndef BYTEPS_COMPRESSOR_IMPL_DRIVE_H
#define BYTEPS_COMPRESSOR_IMPL_DRIVE_H

#include <random>

#include "../compressor.h"
#include "../utils.h"

namespace byteps {
namespace common {
namespace compressor {

class DriveCompressor : public Compressor {
 public:
  DriveCompressor(size_t size, DataType dtype, unsigned int seed = 0)
      : Compressor(size, dtype), _seed(seed) {
        if (seed != 0){
          _rng.set_seed(seed);
        }
      };
  virtual ~DriveCompressor() = default;

  /*!
   * \brief Compress function
   *
   * compress vector and pack into byte array.
   *
   * \param grad gradient tensor
   * \param compressed compressed tensor
   */
  tensor_t Compress(tensor_t grad) override;

  /*!
   * \brief Decompress function
   *
   * unpack from byte array to FP tensor
   *
   * \param compressed compressed tensor
   * \param decompressed decompressed tensor
   */
  tensor_t Decompress(tensor_t compressed) override;

  private:
    //TODO: think about what template should be used
    template <typename index_t, typename scalar_t>
    void HadamardRotate(index_t* dst, const scalar_t* src, size_t len);

    template <typename index_t, typename scalar_t>
    tensor_t CompressImpl(index_t* dst, const scalar_t* src, size_t len);

    template <typename scalar_t, typename index_t>
    tensor_t DecompressImpl(scalar_t* dst, const index_t* src,
                          size_t compressed_size);

  private:
    unsigned int _seed;
    std::random_device _rd;
    XorShift128PlusBitShifterRNG _rng;

};
}  // namespace compressor
}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_COMPRESSOR_IMPL_DRIVE_H