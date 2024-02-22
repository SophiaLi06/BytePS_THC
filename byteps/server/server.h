// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef BYTEPS_SERVER_H
#define BYTEPS_SERVER_H

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <set>
#include <unistd.h>
#include "ps/ps.h"
#include "../common/cpu_reducer.h"
#include "../common/compressor/compressor.h"
#include "../common/compressor/compressor_registry.h"

#ifdef GPU_SERVER
// Enable torch functionality
#include <torch/extension.h>
#include <torch/torch.h>
#endif

namespace byteps {
namespace server {

#define SERVER_KEY_TYPE uint64_t
#define SERVER_DATA_TYPE char
#define DEBUG_PRINT_TENSOR_VALUE(X) (*((float *)(X) + 0))
#define DEBUG_PRINT_TENSOR_ADDRESS(X) (reinterpret_cast<uint64_t>(X))

#define PS_OVERHEAD

using namespace ps;

enum class RequestType {
  kDefaultPushPull, kRowSparsePushPull, kCompressedPushPull, kTopKPushPull, kDGCPushPull, kFindMin, kFindMax, kTHC, kTernGrad
};

enum BytePSEngineOperation {
  SUM_RECV, COPY_FIRST, ALL_RECV, TERMINATE
};

struct PSKV {
  SArray<Key> keys;  // n keys
  SArray<int> lens;  // the length of the i-th value
};

struct DataHandleType {
  RequestType requestType;
  int dtype;
};

struct BytePSArray {
  char* tensor;
  size_t len;
  int dtype;
  ps::KVPairs<char> tmp_sarray;
};

struct UpdateBuf {
  std::vector<ps::KVMeta> request;
  BytePSArray merged;
};

struct BytePSEngineMessage {
  uint64_t id;
  DataHandleType type;
  uint64_t key;
  void* dst;
  void* src;
  size_t len;
  BytePSEngineOperation ops;
  ps::KVPairs<char> sarray; // to temporarily hold it and auto release
  ps::KVMeta req_meta;
};

static DataHandleType DepairDataHandleType(int cmd) {
  int w = std::floor((std::sqrt(8 * cmd + 1) - 1)/2);
  int t = ((w * w) + w) / 2;
  int y = cmd - t;
  int x = w - y;
  CHECK_GE(x, 0);
  CHECK_GE(y, 0);
  DataHandleType type;
  type.requestType = static_cast<RequestType>(x);
  type.dtype = y;
  return type;
}


KVServer<SERVER_DATA_TYPE>* byteps_server_;
byteps::common::CpuReducer* bps_reducer_;

std::mutex pullresp_mu_;
std::unordered_map<uint64_t, ps::KVPairs<char> > push_response_map_;
std::unordered_map<uint64_t, ps::KVPairs<char> > pull_response_map_;

// push & pull flag
std::vector<std::mutex> flag_mu_;
std::vector<std::unordered_map<uint64_t, bool> > is_push_finished_;
std::vector<std::unordered_map<uint64_t, std::vector<ps::KVMeta> > > q_pull_reqmeta_;
std::vector<std::unordered_map<uint64_t, std::set<int> > > seen_sender_;
std::vector<std::unordered_map<uint64_t, size_t> > pull_cnt_;

// byteps handler
std::mutex handle_mu_;
std::mutex update_buf_mu_;
std::unordered_map<uint64_t, UpdateBuf> update_buf_;
std::unordered_map<uint64_t, std::unique_ptr<common::compressor::Compressor>> compressor_map_;

// address map
std::mutex store_mu_;
std::unordered_map<uint64_t, BytePSArray> store_;
std::mutex DGC_store_mu_;
std::unordered_map<uint64_t, BytePSArray> DGC_store_;

// hash function
std::mutex hash_mu_;
std::unordered_map<uint64_t, size_t> hash_cache_;
std::vector<uint64_t> acc_load_; // accumulated tensor size for an engine thread

// global knob
uint64_t timestamp_ = 0;
size_t engine_thread_num_ = 4;
volatile bool is_engine_blocking_ = false;
volatile bool log_key_info_ = false;
volatile bool sync_mode_ = true;
volatile bool debug_mode_ = false;
volatile bool enable_schedule_ = false;

ps::Node::Role role_;
int preferred_rank = -1;
volatile bool is_server_ = true;

// debug
uint64_t debug_key_;
std::mutex debug_mu_;

// new inca table
// std::unordered_map<uint16_t, uint16_t> recv_table = {{0, 0},{1, 3},{2, 5},{3, 7},{4, 9},{5, 11},{6, 13},{7, 14},{8, 16},{9, 17},{10, 19},{11, 21},{12, 23},{13, 25},{14, 27},{15, 30},{256, 768},{257, 771},{258, 773},{259, 775},{260, 777},{261, 779},{262, 781},{263, 782},{264, 784},{265, 785},{266, 787},{267, 789},{268, 791},{269, 793},{270, 795},{271, 798},{512, 1280},{513, 1283},{514, 1285},{515, 1287},{516, 1289},{517, 1291},{518, 1293},{519, 1294},{520, 1296},{521, 1297},{522, 1299},{523, 1301},{524, 1303},{525, 1305},{526, 1307},{527, 1310},{768, 1792},{769, 1795},{770, 1797},{771, 1799},{772, 1801},{773, 1803},{774, 1805},{775, 1806},{776, 1808},{777, 1809},{778, 1811},{779, 1813},{780, 1815},{781, 1817},{782, 1819},{783, 1822},{1024, 2304},{1025, 2307},{1026, 2309},{1027, 2311},{1028, 2313},{1029, 2315},{1030, 2317},{1031, 2318},{1032, 2320},{1033, 2321},{1034, 2323},{1035, 2325},{1036, 2327},{1037, 2329},{1038, 2331},{1039, 2334},{1280, 2816},{1281, 2819},{1282, 2821},{1283, 2823},{1284, 2825},{1285, 2827},{1286, 2829},{1287, 2830},{1288, 2832},{1289, 2833},{1290, 2835},{1291, 2837},{1292, 2839},{1293, 2841},{1294, 2843},{1295, 2846},{1536, 3328},{1537, 3331},{1538, 3333},{1539, 3335},{1540, 3337},{1541, 3339},{1542, 3341},{1543, 3342},{1544, 3344},{1545, 3345},{1546, 3347},{1547, 3349},{1548, 3351},{1549, 3353},{1550, 3355},{1551, 3358},{1792, 3584},{1793, 3587},{1794, 3589},{1795, 3591},{1796, 3593},{1797, 3595},{1798, 3597},{1799, 3598},{1800, 3600},{1801, 3601},{1802, 3603},{1803, 3605},{1804, 3607},{1805, 3609},{1806, 3611},{1807, 3614},{2048, 4096},{2049, 4099},{2050, 4101},{2051, 4103},{2052, 4105},{2053, 4107},{2054, 4109},{2055, 4110},{2056, 4112},{2057, 4113},{2058, 4115},{2059, 4117},{2060, 4119},{2061, 4121},{2062, 4123},{2063, 4126},{2304, 4352},{2305, 4355},{2306, 4357},{2307, 4359},{2308, 4361},{2309, 4363},{2310, 4365},{2311, 4366},{2312, 4368},{2313, 4369},{2314, 4371},{2315, 4373},{2316, 4375},{2317, 4377},{2318, 4379},{2319, 4382},{2560, 4864},{2561, 4867},{2562, 4869},{2563, 4871},{2564, 4873},{2565, 4875},{2566, 4877},{2567, 4878},{2568, 4880},{2569, 4881},{2570, 4883},{2571, 4885},{2572, 4887},{2573, 4889},{2574, 4891},{2575, 4894},{2816, 5376},{2817, 5379},{2818, 5381},{2819, 5383},{2820, 5385},{2821, 5387},{2822, 5389},{2823, 5390},{2824, 5392},{2825, 5393},{2826, 5395},{2827, 5397},{2828, 5399},{2829, 5401},{2830, 5403},{2831, 5406},{3072, 5888},{3073, 5891},{3074, 5893},{3075, 5895},{3076, 5897},{3077, 5899},{3078, 5901},{3079, 5902},{3080, 5904},{3081, 5905},{3082, 5907},{3083, 5909},{3084, 5911},{3085, 5913},{3086, 5915},{3087, 5918},{3328, 6400},{3329, 6403},{3330, 6405},{3331, 6407},{3332, 6409},{3333, 6411},{3334, 6413},{3335, 6414},{3336, 6416},{3337, 6417},{3338, 6419},{3339, 6421},{3340, 6423},{3341, 6425},{3342, 6427},{3343, 6430},{3584, 6912},{3585, 6915},{3586, 6917},{3587, 6919},{3588, 6921},{3589, 6923},{3590, 6925},{3591, 6926},{3592, 6928},{3593, 6929},{3594, 6931},{3595, 6933},{3596, 6935},{3597, 6937},{3598, 6939},{3599, 6942},{3840, 7680},{3841, 7683},{3842, 7685},{3843, 7687},{3844, 7689},{3845, 7691},{3846, 7693},{3847, 7694},{3848, 7696},{3849, 7697},{3850, 7699},{3851, 7701},{3852, 7703},{3853, 7705},{3854, 7707},{3855, 7710}};

// 30 maxval, 16 qlevels, 256 ofreq
// uint8_t recv_table[16] = { 0,  4,  6,  8,  10, 12, 13, 14, 16, 17, 18, 20, 22, 24, 26, 30};
// 30 maxval, 16 qlevels, 32 ofreq
// uint8_t recv_table[16] = { 0,  3,  5,  7,  9, 11, 13, 14, 16, 17, 19, 21, 23, 25, 27, 30};
// uint16_t recv_table[256] = {0, 768, 1280, 1792, 2304, 2816, 3328, 3584, 4096, 4352, 4864, 5376, 5888, 6400, 6912, 7680, 3, 771, 1283, 1795, 2307, 2819, 3331, 3587, 4099, 4355, 4867, 5379, 5891, 6403, 6915, 7683, 5, 773, 1285, 1797, 2309, 2821, 3333, 3589, 4101, 4357, 4869, 5381, 5893, 6405, 6917, 7685, 7, 775, 1287, 1799, 2311, 2823, 3335, 3591, 4103, 4359, 4871, 5383, 5895, 6407, 6919, 7687, 9, 777, 1289, 1801, 2313, 2825, 3337, 3593, 4105, 4361, 4873, 5385, 5897, 6409, 6921, 7689, 11, 779, 1291, 1803, 2315, 2827, 3339, 3595, 4107, 4363, 4875, 5387, 5899, 6411, 6923, 7691, 13, 781, 1293, 1805, 2317, 2829, 3341, 3597, 4109, 4365, 4877, 5389, 5901, 6413, 6925, 7693, 14, 782, 1294, 1806, 2318, 2830, 3342, 3598, 4110, 4366, 4878, 5390, 5902, 6414, 6926, 7694, 16, 784, 1296, 1808, 2320, 2832, 3344, 3600, 4112, 4368, 4880, 5392, 5904, 6416, 6928, 7696, 17, 785, 1297, 1809, 2321, 2833, 3345, 3601, 4113, 4369, 4881, 5393, 5905, 6417, 6929, 7697, 19, 787, 1299, 1811, 2323, 2835, 3347, 3603, 4115, 4371, 4883, 5395, 5907, 6419, 6931, 7699, 21, 789, 1301, 1813, 2325, 2837, 3349, 3605, 4117, 4373, 4885, 5397, 5909, 6421, 6933, 7701, 23, 791, 1303, 1815, 2327, 2839, 3351, 3607, 4119, 4375, 4887, 5399, 5911, 6423, 6935, 7703, 25, 793, 1305, 1817, 2329, 2841, 3353, 3609, 4121, 4377, 4889, 5401, 5913, 6425, 6937, 7705, 27, 795, 1307, 1819, 2331, 2843, 3355, 3611, 4123, 4379, 4891, 5403, 5915, 6427, 6939, 7707, 30, 798, 1310, 1822, 2334, 2846, 3358, 3614, 4126, 4382, 4894, 5406, 5918, 6430, 6942, 7710};

// PS overhead timing
#ifdef PS_OVERHEAD
std::unordered_map<uint64_t, std::chrono::time_point<std::chrono::system_clock>> PS_start;
std::chrono::time_point<std::chrono::system_clock> Compress_start;
double total_PS_time = 0;
int total_PS_time_points = 0;
double total_compress_time = 0;
#endif

int DivUp(int x, int y) { return (x + y - 1) / y; }
int RoundUp(int x, int y) { return DivUp(x, y) * y; }

uint64_t DecodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key - kr.begin();
}

uint64_t EncodeKey(ps::Key key) {
  auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
  return key + kr.begin();
}

size_t GetThreadID(uint64_t key, size_t len) {
  std::lock_guard<std::mutex> lock(hash_mu_);
  if (len == 0) { // pull
    CHECK_NE(hash_cache_.find(key), hash_cache_.end());
    return hash_cache_[key];
  }
  if (hash_cache_.find(key) != hash_cache_.end()) {
    return hash_cache_[key];
  }
  CHECK_GT(len, 0);
  CHECK_EQ(acc_load_.size(), engine_thread_num_);
  auto min_index = -1;
  auto min_load = std::numeric_limits<uint64_t>::max();
  for (size_t i = 0; i < engine_thread_num_; ++i) {
    if (acc_load_[i] < min_load) {
      min_load = acc_load_[i];
      min_index = i;
    }
  }
  CHECK_GE(min_index, 0);
  CHECK_LT(min_index, engine_thread_num_);
  acc_load_[min_index] += len;
  hash_cache_[key] = min_index;
  return hash_cache_[key];
}

void PageAlignedMalloc(void** ptr, size_t size) {
  size_t page_size = sysconf(_SC_PAGESIZE);
  void* p;
  int size_aligned = RoundUp(size, page_size);
  int ret = posix_memalign(&p, page_size, size_aligned);
  CHECK_EQ(ret, 0) << "posix_memalign error: " << strerror(ret);
  CHECK(p);
  memset(p, 0, size);
  *ptr = p;
}

extern "C" void byteps_server();

}  // namespace server
}  // namespace byteps

#endif  // BYTEPS_SERVER_H
