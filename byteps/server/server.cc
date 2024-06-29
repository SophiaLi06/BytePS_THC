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

#include "server.h"
#include "../common/compressor/utils.h"
#include "queue.h"

namespace byteps {
namespace server {

using namespace ps;

// engine related
std::vector<PriorityQueue*> engine_queues_;
std::vector<std::thread*> engine_threads_;

BytePSArray* GetStore(uint64_t key) {
  std::lock_guard<std::mutex> lock(store_mu_);
  return &store_[key];
}

BytePSArray* GetDGCStore(uint64_t key) {
  std::lock_guard<std::mutex> lock(DGC_store_mu_);
  return &DGC_store_[key];
}

UpdateBuf* GetUpdateBuf(uint64_t key) {
  std::lock_guard<std::mutex> lock(update_buf_mu_);
  return &update_buf_[key];
} 

#ifdef PS_OVERHEAD
void RecordPacketTimestamp(uint64_t key) {
  PS_start[key]= std::chrono::high_resolution_clock::now();
}

std::chrono::time_point<std::chrono::system_clock> GetPacketTimestamp(uint64_t key) {
  return PS_start[key];
}
#endif

void SendPushResponse(uint64_t key, const ps::KVMeta& req,
                      ps::KVServer<char>* server) {
  auto iterator = push_response_map_.find(key);
  if (iterator == push_response_map_.end()) {  // new key
    ps::KVPairs<char> response;
    push_response_map_[key] = response;  // add to the map
    server->Response(req, response);
  } else {  // not new key, then reuse the memory address to avoid ibv_reg_mr on
            // RDMA data path
    ps::KVPairs<char>* response = &iterator->second;
    server->Response(req, *response);
  }
}

void SendPullResponse(const DataHandleType type, const uint64_t key,
                      const ps::KVMeta& req_meta, ps::KVServer<char>* server) {
  std::lock_guard<std::mutex> lock(pullresp_mu_);
  auto updates = GetUpdateBuf(key);
  CHECK(updates->merged.tensor) << "init " << key << " first";
  char* data = updates->merged.tensor;
  auto len = updates->merged.len;
  // printf("server pull response update len %d\n", len);

  // send pull response
  auto iterator = pull_response_map_.find(key);
  if (iterator == pull_response_map_.end()) {  // new key
    ps::KVPairs<char> response;
    response.keys = {EncodeKey(key)};
    response.lens = {len};
    response.vals = ps::SArray<char>(data, len, false);  // zero copy
    pull_response_map_[key] = response;                  // add to the map
    server->Response(req_meta, response);
  } else {  // not new key, then reuse the memory address to avoid ibv_reg_mr on
            // RDMA data path
    ps::KVPairs<char>* response = &iterator->second;

    auto p = static_cast<char*>(data);
    CHECK(p);
    response->lens = {len};
    response->vals = ps::SArray<char>(p, len, false);
    server->Response(req_meta, *response);
  }
  // uint8_t* data_uint8 = (uint8_t*)data;
  // for (int i = 0; i < 10; ++i) printf("%hhu ", data_uint8[i]);
  // printf("\n");
}

void BytePSServerEngineThread(int i) {
  auto& q = engine_queues_[i];
  while (true) {
    BytePSEngineMessage msg;
    q->WaitAndPop(&msg);
    if (msg.ops == TERMINATE) break;
    // do some check
    CHECK(msg.dst);
    CHECK(msg.src);

    auto iter = compressor_map_.find(msg.key);
    if (iter != compressor_map_.end()) {
      // compress
      if (msg.ops == ALL_RECV) {
        common::compressor::tensor_t grad(reinterpret_cast<char*>(msg.src),
                                          msg.len, msg.type.dtype);
        auto compressed = iter->second->Compress(grad);
        // 1. BytePS built-in compress
        auto updates = GetUpdateBuf(msg.key);
        updates->merged.tensor = compressed.data;
        updates->merged.len = compressed.size;
      } else {  // decompress
        auto compressed_len = msg.sarray.lens[0];
        CHECK_LE(compressed_len, msg.len);
        common::compressor::tensor_t compressed(
            reinterpret_cast<char*>(msg.src), compressed_len, msg.type.dtype);
        auto decompressed = iter->second->Decompress(compressed);
        msg.src = decompressed.data;
      }
    } else {
      if (msg.ops == ALL_RECV) {
        if(msg.type.requestType == RequestType::kTopKPushPull
          || msg.type.requestType == RequestType::kDGCPushPull){
#ifdef PS_OVERHEAD
          auto compression_start = std::chrono::high_resolution_clock::now();
#endif
          auto stored = GetStore(msg.key);
          // Vector to store element with respective present index
          std::vector<std::pair<float, int> > vp;
          auto half_len = (msg.len / sizeof(float)) / 2;
          // Note that here msg.dst and msg.src are both stored->tensor
          float* msg_data_ptr = (float*)msg.src;
          auto topk_comp = [](std::pair<float, int> &a, std::pair<float, int> &b) {
            return std::abs(a.first) > std::abs(b.first);
          };
          for (int i = 0; i < half_len; ++i)
          {
            vp.push_back(std::make_pair(msg_data_ptr[i], i));
          }
          std::make_heap(vp.begin(), vp.end(), topk_comp);
          for (int i = half_len; i < (stored->len) / sizeof(float); ++i){
            if (std::abs(msg_data_ptr[i]) > std::abs(vp.front().first)){
              std::pop_heap(vp.begin(), vp.end(), topk_comp);
              vp.back() = std::make_pair(msg_data_ptr[i], i);
              std::push_heap(vp.begin(), vp.end(), topk_comp);
            }
          }
          if (msg.type.requestType == RequestType::kDGCPushPull){
            auto DGC_stored = GetDGCStore(msg.key);
            memcpy((float*)(DGC_stored->tensor), msg_data_ptr, stored->len);
            float* DGC_stored_ptr = (float*)(DGC_stored->tensor);
            for (int i = 0; i < half_len; ++i){
              // equivalent to apply ~Mask to sent coordinates
              DGC_stored_ptr[vp[i].second] = 0;
            }
          }
          for (int i = 0; i < half_len; ++i){
            msg_data_ptr[i] = vp[i].second;
            msg_data_ptr[half_len+i] = vp[i].first;
            // printf("idx %d(%.2f, %.6f) ", i, msg_data_ptr[i], msg_data_ptr[half_len+i]);
          }
          // printf("\n");
#ifdef PS_OVERHEAD
          std::chrono::duration<double, std::milli> time_span =
            (std::chrono::high_resolution_clock::now() - compression_start);
          total_compress_time += time_span.count();
#endif
        }
        // 2. no BytePS built-in compress, update the merged.tensor using msg.src here
        auto updates = GetUpdateBuf(msg.key);
        updates->merged.tensor = reinterpret_cast<char*>(msg.src);
        updates->merged.len = msg.len;
        if(bps_reducer_->GetDataType(msg.type.dtype) == common::BYTEPS_UINT8 && 
             msg.type.requestType == RequestType::kTHC) updates->merged.len *= 2;
        else if (bps_reducer_->GetDataType(msg.type.dtype) == common::BYTEPS_UINT8 && 
             msg.type.requestType == RequestType::kTernGrad) updates->merged.len *= 4;
      }
    }

    bool is_debug = (debug_mode_ && (debug_key_ == msg.key));
    switch (msg.ops) {
      case COPY_FIRST: {
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_BEFORE \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
        if(msg.type.requestType == RequestType::kTopKPushPull 
          || msg.type.requestType == RequestType::kDGCPushPull){
          // printf("received first kTopK Push request of key %d and len %d\n", 
          //   msg.key, msg.len);
#ifdef PS_OVERHEAD
          auto decompression_start = std::chrono::high_resolution_clock::now();
#endif
          float* msg_data_ptr = (float*)msg.src;
          float* msg_dst_ptr = (float*)msg.dst;
          auto stored = GetStore(msg.key);
          if (msg.type.requestType == RequestType::kDGCPushPull){
            // add DGC stored coordinates values
            auto DGC_stored = GetDGCStore(msg.key);
            if (!DGC_stored->tensor) {
              // init DGC stored buffer, use page aligned memory
              size_t aligned_size = common::Align(stored->len, msg.type.dtype);
              PageAlignedMalloc((void**)&DGC_stored->tensor, aligned_size);
              memset((float*)(DGC_stored->tensor), 0, stored->len);
            }
            memcpy(msg_dst_ptr, (float*)(DGC_stored->tensor), stored->len);
          }
          else memset(msg_dst_ptr, 0, stored->len);
          auto half_len = (msg.len / sizeof(float)) / 2;
// #pragma omp parallel for simd
// #pragma omp parallel for simd num_threads(engine_thread_num_)
#ifdef PS_OVERHEAD
          auto topk_agg_start = std::chrono::high_resolution_clock::now();
#endif
          for (int i = 0; i < half_len; ++i){
            msg_dst_ptr[int(msg_data_ptr[i])] += msg_data_ptr[half_len+i];
          }
#ifdef PS_OVERHEAD
          std::chrono::duration<double, std::milli> topk_agg_span = (std::chrono::high_resolution_clock::now() - topk_agg_start);
          total_PS_time += topk_agg_span.count();
          total_PS_time_points += 1;
#endif
#ifdef PS_OVERHEAD
          std::chrono::duration<double, std::milli> time_span =
            (std::chrono::high_resolution_clock::now() - decompression_start);
          total_compress_time += time_span.count();
#endif
        }
        else{
#ifdef PS_OVERHEAD
          auto agg_start = std::chrono::high_resolution_clock::now();
#endif
          if(bps_reducer_->GetDataType(msg.type.dtype) == common::BYTEPS_UINT8 && 
             msg.type.requestType == RequestType::kTHC){
            // uint8_t recv_table[16] = { 0,  3,  5,  7,  9, 11, 13, 14, 16, 17, 19, 21, 23, 25, 27, 30};
            uint16_t recv_table[256] = {0, 768, 1280, 1792, 2304, 2816, 3328, 3584, 4096, 4352, 4864, 5376, 5888, 6400, 6912, 7680, 3, 771, 1283, 1795, 2307, 2819, 3331, 3587, 4099, 4355, 4867, 5379, 5891, 6403, 6915, 7683, 5, 773, 1285, 1797, 2309, 2821, 3333, 3589, 4101, 4357, 4869, 5381, 5893, 6405, 6917, 7685, 7, 775, 1287, 1799, 2311, 2823, 3335, 3591, 4103, 4359, 4871, 5383, 5895, 6407, 6919, 7687, 9, 777, 1289, 1801, 2313, 2825, 3337, 3593, 4105, 4361, 4873, 5385, 5897, 6409, 6921, 7689, 11, 779, 1291, 1803, 2315, 2827, 3339, 3595, 4107, 4363, 4875, 5387, 5899, 6411, 6923, 7691, 13, 781, 1293, 1805, 2317, 2829, 3341, 3597, 4109, 4365, 4877, 5389, 5901, 6413, 6925, 7693, 14, 782, 1294, 1806, 2318, 2830, 3342, 3598, 4110, 4366, 4878, 5390, 5902, 6414, 6926, 7694, 16, 784, 1296, 1808, 2320, 2832, 3344, 3600, 4112, 4368, 4880, 5392, 5904, 6416, 6928, 7696, 17, 785, 1297, 1809, 2321, 2833, 3345, 3601, 4113, 4369, 4881, 5393, 5905, 6417, 6929, 7697, 19, 787, 1299, 1811, 2323, 2835, 3347, 3603, 4115, 4371, 4883, 5395, 5907, 6419, 6931, 7699, 21, 789, 1301, 1813, 2325, 2837, 3349, 3605, 4117, 4373, 4885, 5397, 5909, 6421, 6933, 7701, 23, 791, 1303, 1815, 2327, 2839, 3351, 3607, 4119, 4375, 4887, 5399, 5911, 6423, 6935, 7703, 25, 793, 1305, 1817, 2329, 2841, 3353, 3609, 4121, 4377, 4889, 5401, 5913, 6425, 6937, 7705, 27, 795, 1307, 1819, 2331, 2843, 3355, 3611, 4123, 4379, 4891, 5403, 5915, 6427, 6939, 7707, 30, 798, 1310, 1822, 2334, 2846, 3358, 3614, 4126, 4382, 4894, 5406, 5918, 6430, 6942, 7710};
            
            uint8_t* msg_src_ptr = (uint8_t*)msg.src;
            auto stored = GetStore(msg.key);
            uint16_t* msg_data = (uint16_t*)msg.dst;
            // printf("COPY_FIRST UINT8 tensor of key %d, sent length %d, stored length %d\n",
            //   msg.key, msg.len, stored->len);
// #pragma omp parallel for simd
#pragma omp parallel for simd num_threads(4)
            for (size_t i = 0; i < msg.len; i++) {
              // msg_src_ptr[i] = recv_table[msg_src_ptr[i]];
              msg_data[i] = recv_table[msg_src_ptr[i]];
            }
          }
          else if(bps_reducer_->GetDataType(msg.type.dtype) == common::BYTEPS_UINT8 && 
             msg.type.requestType == RequestType::kTernGrad){
            auto stored = GetStore(msg.key);
            uint8_t* msg_src_ptr = (uint8_t*)msg.src;
            uint8_t* msg_data = (uint8_t*)msg.dst;
            // printf("COPY_FIRST UINT8 tensor of key %d, sent length %d, stored length %d\n",
            //   msg.key, msg.len, stored->len);
#pragma omp parallel for simd num_threads(4)
            for (size_t i = 0; i < msg.len; i++) {
              msg_data[4*i] = msg_src_ptr[i]>>6;
              msg_data[4*i+1] = (msg_src_ptr[i] & 0b00110000)>>4;
              msg_data[4*i+2] = (msg_src_ptr[i] & 0b00001100)>>2;
              msg_data[4*i+3] = msg_src_ptr[i] & 0b00000011;
            }
          }
          // bps_reducer_->copy(msg.dst, msg.src, msg.len);
          else bps_reducer_->copy(msg.dst, msg.src, msg.len);
#ifdef PS_OVERHEAD
          std::chrono::duration<double, std::milli> agg_span = (std::chrono::high_resolution_clock::now() - agg_start);
          total_PS_time += agg_span.count();
          total_PS_time_points += 1;
#endif
        }
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_COPY_MERGED_TO_STORE_AFTER \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
      } break;

      case ALL_RECV: {
        std::lock_guard<std::mutex> lock(flag_mu_[i]);
        if (is_push_finished_[i].find(msg.key) == is_push_finished_[i].end()) {
          is_push_finished_[i][msg.key] = false;
          pull_cnt_[i][msg.key] = 0;
          seen_sender_[i][msg.key].clear();
        }
        is_push_finished_[i][msg.key] = true;

        auto it = q_pull_reqmeta_[i][msg.key].begin();
        while (it != q_pull_reqmeta_[i][msg.key].end()) {
          if (seen_sender_[i][msg.key].find(it->sender) ==
              seen_sender_[i][msg.key].end()) {
            SendPullResponse(msg.type, msg.key, *it, byteps_server_);
            pull_cnt_[i][msg.key] += 1;
            seen_sender_[i][msg.key].insert(it->sender);
            it = q_pull_reqmeta_[i][msg.key].erase(it);
          } else {
            ++it;
          }
          if (pull_cnt_[i][msg.key] == (size_t)ps::NumWorkers()) {
            is_push_finished_[i][msg.key] = false;
            pull_cnt_[i][msg.key] = 0;
            seen_sender_[i][msg.key].clear();
            break;
          }
        }
// #ifdef PS_OVERHEAD
//         auto current_time = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> time_span =
//             (current_time - GetPacketTimestamp(msg.key));
//         total_PS_time += time_span.count();
// #endif
      } break;

      case SUM_RECV: {
        auto bps_type = bps_reducer_->GetDataType(msg.type.dtype);
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_BEFORE \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
        if(msg.type.requestType == RequestType::kTopKPushPull
          || msg.type.requestType == RequestType::kDGCPushPull){
          // printf("received following kTopK Push request of key %d and len %d\n", 
          //   msg.key, msg.len);
#ifdef PS_OVERHEAD
          auto decompression_start = std::chrono::high_resolution_clock::now();
#endif
          float* msg_data_ptr = (float*)msg.src;
          float* msg_dst_ptr = (float*)msg.dst;
          auto half_len = (msg.len / sizeof(float)) / 2;
// #pragma omp parallel for simd
// #pragma omp parallel for simd num_threads(engine_thread_num_)
#ifdef PS_OVERHEAD
          auto topk_sum_recv_start = std::chrono::high_resolution_clock::now();
#endif
          for (int i = 0; i < half_len; ++i){
            msg_dst_ptr[int(msg_data_ptr[i])] += msg_data_ptr[half_len+i];
          }
#ifdef PS_OVERHEAD
          std::chrono::duration<double, std::milli> topk_sum_recv_span = (std::chrono::high_resolution_clock::now() - topk_sum_recv_start);
          total_PS_time += topk_sum_recv_span.count();
          total_PS_time_points += 1;
#endif
#ifdef PS_OVERHEAD
          std::chrono::duration<double, std::milli> time_span =
            (std::chrono::high_resolution_clock::now() - decompression_start);
          total_compress_time += time_span.count();
#endif
        }
        else{
#ifdef PS_OVERHEAD
          auto sum_recv_start = std::chrono::high_resolution_clock::now();
#endif
          if(bps_type == common::BYTEPS_UINT8 && 
             msg.type.requestType == RequestType::kTHC){
            // uint8_t recv_table[16] = { 0,  3,  5,  7,  9, 11, 13, 14, 16, 17, 19, 21, 23, 25, 27, 30};
            uint16_t recv_table[256] = {0, 768, 1280, 1792, 2304, 2816, 3328, 3584, 4096, 4352, 4864, 5376, 5888, 6400, 6912, 7680, 3, 771, 1283, 1795, 2307, 2819, 3331, 3587, 4099, 4355, 4867, 5379, 5891, 6403, 6915, 7683, 5, 773, 1285, 1797, 2309, 2821, 3333, 3589, 4101, 4357, 4869, 5381, 5893, 6405, 6917, 7685, 7, 775, 1287, 1799, 2311, 2823, 3335, 3591, 4103, 4359, 4871, 5383, 5895, 6407, 6919, 7687, 9, 777, 1289, 1801, 2313, 2825, 3337, 3593, 4105, 4361, 4873, 5385, 5897, 6409, 6921, 7689, 11, 779, 1291, 1803, 2315, 2827, 3339, 3595, 4107, 4363, 4875, 5387, 5899, 6411, 6923, 7691, 13, 781, 1293, 1805, 2317, 2829, 3341, 3597, 4109, 4365, 4877, 5389, 5901, 6413, 6925, 7693, 14, 782, 1294, 1806, 2318, 2830, 3342, 3598, 4110, 4366, 4878, 5390, 5902, 6414, 6926, 7694, 16, 784, 1296, 1808, 2320, 2832, 3344, 3600, 4112, 4368, 4880, 5392, 5904, 6416, 6928, 7696, 17, 785, 1297, 1809, 2321, 2833, 3345, 3601, 4113, 4369, 4881, 5393, 5905, 6417, 6929, 7697, 19, 787, 1299, 1811, 2323, 2835, 3347, 3603, 4115, 4371, 4883, 5395, 5907, 6419, 6931, 7699, 21, 789, 1301, 1813, 2325, 2837, 3349, 3605, 4117, 4373, 4885, 5397, 5909, 6421, 6933, 7701, 23, 791, 1303, 1815, 2327, 2839, 3351, 3607, 4119, 4375, 4887, 5399, 5911, 6423, 6935, 7703, 25, 793, 1305, 1817, 2329, 2841, 3353, 3609, 4121, 4377, 4889, 5401, 5913, 6425, 6937, 7705, 27, 795, 1307, 1819, 2331, 2843, 3355, 3611, 4123, 4379, 4891, 5403, 5915, 6427, 6939, 7707, 30, 798, 1310, 1822, 2334, 2846, 3358, 3614, 4126, 4382, 4894, 5406, 5918, 6430, 6942, 7710};
            
            uint8_t* msg_src_ptr = (uint8_t*)msg.src;
            auto stored = GetStore(msg.key);
            uint16_t* msg_data = (uint16_t*)msg.dst;
            // printf("SUM_RECV UINT8 tensor of key %d, sent length %d, stored length %d\n",
            //   msg.key, msg.len, stored->len);
// #pragma omp parallel for simd
#pragma omp parallel for simd num_threads(4)
            for (size_t i = 0; i < msg.len; i++) {
              // msg_src_ptr[i] = recv_table[msg_src_ptr[i]];
              msg_data[i] += recv_table[msg_src_ptr[i]];
            }
          }
          else if(bps_type == common::BYTEPS_UINT8 && 
             msg.type.requestType == RequestType::kTernGrad){
            auto stored = GetStore(msg.key);
            uint8_t* msg_src_ptr = (uint8_t*)msg.src;
            uint8_t* msg_data = (uint8_t*)msg.dst;
            // printf("SUM_RECV UINT8 tensor of key %d, sent length %d, stored length %d\n",
            //   msg.key, msg.len, stored->len);
#pragma omp parallel for simd num_threads(4)
            for (size_t i = 0; i < msg.len; i++) {
              msg_data[4*i] += msg_src_ptr[i]>>6;
              msg_data[4*i+1] += (msg_src_ptr[i] & 0b00110000)>>4;
              msg_data[4*i+2] += (msg_src_ptr[i] & 0b00001100)>>2;
              msg_data[4*i+3] += msg_src_ptr[i] & 0b00000011;
            }
          }
          // CHECK_GE(bps_reducer_->sum(msg.dst, msg.src, msg.len, bps_type), 0);
          else CHECK_GE(bps_reducer_->sum(msg.dst, msg.src, msg.len, bps_type), 0);
#ifdef PS_OVERHEAD
          std::chrono::duration<double, std::milli> sum_recv_span = (std::chrono::high_resolution_clock::now() - sum_recv_start);
          total_PS_time += sum_recv_span.count();
          total_PS_time_points += 1;
#endif
        }
        if (is_debug) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: ENGINE_SUM_RECV_AFTER \t"
                    << "dst: " << DEBUG_PRINT_TENSOR_VALUE(msg.dst) << "\t"
                    << "src: " << DEBUG_PRINT_TENSOR_VALUE(msg.src) << "\t"
                    << "dst_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.dst)
                    << "\t"
                    << "src_addr: " << DEBUG_PRINT_TENSOR_ADDRESS(msg.src)
                    << "\t";
        }
      } break;
      default:
        CHECK(0);
    }
  }
}  // namespace server

void BytePSHandler(const ps::KVMeta& req_meta,
                   const ps::KVPairs<char>& req_data,
                   ps::KVServer<char>* server) {
  std::lock_guard<std::mutex> lock(handle_mu_);  // push & pull may have racing
  DataHandleType type = DepairDataHandleType(req_meta.cmd);
  // CHECK_EQ(type.requestType, RequestType::kDefaultPushPull);
  // do some check
  CHECK_EQ(req_data.keys.size(), (size_t)1);
  if (log_key_info_) {
    if (req_meta.push) {
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
      LOG(INFO) << "push key=" << DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender
                << "\t size=" << (size_t)req_data.lens[0];
    } else {
      LOG(INFO) << "pull key=" << (uint64_t)DecodeKey(req_data.keys[0])
                << "\t sender=" << req_meta.sender;
    }
  }
  uint64_t key = DecodeKey(req_data.keys[0]);

  // register compressor
  if (type.requestType == RequestType::kCompressedPushPull) {
    if (compressor_map_.find(key) == compressor_map_.end()) {
      std::string content{reinterpret_cast<char*>(req_data.vals.data()),
                          static_cast<size_t>(req_data.lens[0])};
      auto kwargs = byteps::common::compressor::Deserialize(content);
      auto stored = GetStore(key);
      size_t aligned_size = byteps::common::Align(stored->len, stored->dtype);
      auto compressor_ptr =
          byteps::common::compressor::CompressorRegistry::Create(
              kwargs, aligned_size,
              static_cast<byteps::common::DataType>(stored->dtype));
      CHECK_NE(compressor_ptr, nullptr);
      compressor_map_[key] = std::move(compressor_ptr);
      if (log_key_info_) {
        LOG(INFO) << "register compressor for key=" << key;
      }
    }

    // buffer the request meta
    auto updates = GetUpdateBuf(key);
    updates->request.push_back(req_meta);
    // should send response after collecting all init push
    if (updates->request.size() < (size_t)ps::NumWorkers()) return;

    for (const auto& req : updates->request) {
      SendPushResponse(key, req, server);
    }
    updates->request.clear();
    return;
  }

  // // process global min/max or max_norm information
  // if (type.requestType == RequestType::kFindMax){
  //   printf("recieved kFindMax request\n");
  // }

  // if (type.requestType == RequestType::kFindMin){
  //   printf("recieved kFindMin request\n");
  // }

  if (req_meta.push) {  // push request
    // CHECK_EQ(req_data.lens.size(), (size_t)1);
    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    auto stored = GetStore(key);
    auto len = (size_t)req_data.lens[0];
    auto recved = reinterpret_cast<char*>(req_data.vals.data());

    if (!stored->tensor) { //initialize the buffer
      auto updates = GetUpdateBuf(key);
      if (sync_mode_) {
        updates->merged.len = len;
        updates->merged.dtype = type.dtype;
      }
      // buffer the request meta
      updates->request.push_back(req_meta);
      // should send response after collecting all init push
      if (updates->request.size() < (size_t)ps::NumWorkers()) return;
      if (log_key_info_) {
        LOG(INFO) << "Collected all " << updates->request.size()
                  << " requests for key=" << key
                  << ", init the store buffer size="
                  << (size_t)req_data.lens[0];
      }
      // init stored buffer, use page aligned memory
      size_t aligned_size = common::Align(len, type.dtype);
      PageAlignedMalloc((void**)&stored->tensor, aligned_size);
      stored->len = len;
      stored->dtype = type.dtype;
      CHECK(stored->tensor);

      bps_reducer_->copy(stored->tensor, recved,
                         len);  // we may not need this copy
      for (const auto& req : updates->request) {
        SendPushResponse(key, req, server);
      }
      updates->request.clear();
    } else {
      auto updates = GetUpdateBuf(key);
      auto tid = GetThreadID(key, len);
      if (updates->request.empty()) {  // from the first incoming worker
// #ifdef PS_OVERHEAD
//         RecordPacketTimestamp(key);
// #endif
        if (sync_mode_) {
          if (debug_mode_ && (debug_key_ == key)) {
            std::lock_guard<std::mutex> lock(debug_mu_);
            LOG(INFO) << "stage: FIRST_WORKER_RECV \t"
                      << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                      << "\t"
                      << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                      << "len: " << len << "\t"
                      << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
          }
          updates->merged.tmp_sarray = req_data;
          // copy
          BytePSEngineMessage msg = {timestamp_++,   type,     key,
                                     stored->tensor, recved,   len,
                                    //  stored->tensor, recved,   stored->len,
                                     COPY_FIRST,     req_data, req_meta};
          engine_queues_[tid]->Push(msg);
        } else {  // async mode, directly add to the buffer
          CHECK_GE(bps_reducer_->sum((void*)stored->tensor, (void*)recved, len,
                                     bps_reducer_->GetDataType(stored->dtype)),
                   0);
        }
      } else {  // from other workers
        CHECK(sync_mode_);
        // CHECK(updates.merged.tensor);
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: OTHER_WORKER_SUM \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                    << "\t"
                    << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved) << "\t"
                    << "len: " << len << "\t"
                    << "addr: " << DEBUG_PRINT_TENSOR_ADDRESS(recved);
        }
        if (is_engine_blocking_) {
          // TODO: decompress
          CHECK_GE(bps_reducer_->sum(
                       (void*)updates->merged.tensor, (void*)recved, len,
                       bps_reducer_->GetDataType(updates->merged.dtype)),
                   0);
        } else {  // non-blocking
          BytePSEngineMessage msg = {timestamp_++,   type,     key,
                                     stored->tensor, recved,   len,
                                    //  stored->tensor, recved,   stored->len,
                                     SUM_RECV,       req_data, req_meta};
          engine_queues_[tid]->Push(msg);
        }
      }
      // add a worker information (request.size() is the # workers received)
      updates->request.push_back(req_meta);
      SendPushResponse(key, req_meta, server);
      if (sync_mode_ && updates->request.size() == (size_t)ps::NumWorkers()) {
        auto stored = GetStore(key);
        auto& update = updates->merged;
        if (debug_mode_ && (debug_key_ == key)) {
          std::lock_guard<std::mutex> lock(debug_mu_);
          LOG(INFO) << "stage: COPY_MERGED_TO_STORE \t"
                    << "stored: " << DEBUG_PRINT_TENSOR_VALUE(stored->tensor)
                    << "\t"
                    << "merged: "
                    << DEBUG_PRINT_TENSOR_VALUE(updates->merged.tensor) << "\t"
                    << "recved: " << DEBUG_PRINT_TENSOR_VALUE(recved);
        }
        if (is_engine_blocking_) {
          // TODO: compress
          bps_reducer_->copy(stored->tensor, updates->merged.tensor, len);
        } else {
          BytePSEngineMessage msg = {
              timestamp_++,   type,        key,     stored->tensor,
              stored->tensor, len, ALL_RECV};
              // stored->tensor, stored->len, ALL_RECV};
          engine_queues_[tid]->Push(msg);
          engine_queues_[tid]->ClearCounter(key);
        }
        updates->request.clear();
      } else if (!sync_mode_) {
        // async: clean the request buffer
        updates->request.clear();
      }
    }
  } else {  // pull request
    auto stored = GetStore(key);
    CHECK(stored->tensor) << "Should init the buffer for key=" << key
                          << " first";
    if (is_engine_blocking_ || !sync_mode_) {
      SendPullResponse(type, key, req_meta, server);
    } else {
      auto tid = GetThreadID(key, 0);
      std::lock_guard<std::mutex> lock(flag_mu_[tid]);
      if (is_push_finished_[tid].find(key) == is_push_finished_[tid].end()) {
        is_push_finished_[tid][key] = false;
        pull_cnt_[tid][key] = 0;
        seen_sender_[tid][key].clear();
      }

      auto it = seen_sender_[tid][key].find(req_meta.sender);
      if (is_push_finished_[tid][key] && (it == seen_sender_[tid][key].end())) {
        // push already finished && not received the associated pull response
        // yet
        SendPullResponse(type, key, req_meta, server);
        pull_cnt_[tid][key] += 1;
        seen_sender_[tid][key].insert(req_meta.sender);

        if (pull_cnt_[tid][key] == (size_t)ps::NumWorkers()) {
          is_push_finished_[tid][key] = false;
          pull_cnt_[tid][key] = 0;
          seen_sender_[tid][key].clear();
        }
      } else {
        // push not finished, put into the queue, and wait for the engine
        q_pull_reqmeta_[tid][key].push_back(req_meta);
      }
    }
  }
}

void init_global_env() {
  // enable to print key profile
  log_key_info_ = GetEnv("PS_KEY_LOG", 0);

  std::string role_str = GetEnv("DMLC_ROLE", "server");
  role_ = ps::GetRole(role_str);
  if (role_str == std::string("server")) {
    is_server_ = true;
    preferred_rank = -1;
  } else {
    is_server_ = false;
    preferred_rank = 0;
  }

  LOG(INFO) << "This is a " << role_str << " is_server=" << is_server_;

  // enable engine block mode (default disabled)
  is_engine_blocking_ = GetEnv("BYTEPS_SERVER_ENGINE_BLOCKING", 0);
  if (is_engine_blocking_)
    LOG(INFO) << "Enable blocking mode of the server engine";

  // sync or async training
  sync_mode_ = !GetEnv("BYTEPS_ENABLE_ASYNC", 0);
  if (!sync_mode_)
    LOG(INFO) << "BytePS server is enabled asynchronous training";

  // debug mode
  debug_mode_ = GetEnv("BYTEPS_SERVER_DEBUG", 0);
  debug_key_ = GetEnv("BYTEPS_SERVER_DEBUG_KEY", 0);
  if (debug_mode_)
    LOG(INFO) << "Debug mode enabled! Printing key " << debug_key_;

  // number of engine thread
  // invalid if is_engine_blocking = true
  engine_thread_num_ = GetEnv("BYTEPS_SERVER_ENGINE_THREAD", 4);
  LOG(INFO) << "BytePS server engine uses " << engine_thread_num_ << " threads"
            << ", consider increasing BYTEPS_SERVER_ENGINE_THREAD for higher "
               "performance";
  CHECK_GE(engine_thread_num_, 1);

  // enable scheduling for server engine
  enable_schedule_ = GetEnv("BYTEPS_SERVER_ENABLE_SCHEDULE", 0);
  if (enable_schedule_)
    LOG(INFO) << "Enable engine scheduling for BytePS server";
}

extern "C" void byteps_server() {
  init_global_env();

  // cpu reducer
  bps_reducer_ = new byteps::common::CpuReducer(nullptr);

  // flag mu and its protected map
  std::vector<std::mutex> tmp_flagmu(engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, bool> > tmp_ispushfinished(
      engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, std::vector<ps::KVMeta> > >
      tmp_qpullreqmeta(engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, std::set<int> > > tmp_seensender(
      engine_thread_num_);
  std::vector<std::unordered_map<uint64_t, size_t> > tmp_pullcnt(
      engine_thread_num_);
  flag_mu_.swap(tmp_flagmu);
  is_push_finished_.swap(tmp_ispushfinished);
  q_pull_reqmeta_.swap(tmp_qpullreqmeta);
  seen_sender_.swap(tmp_seensender);
  pull_cnt_.swap(tmp_pullcnt);
  CHECK_EQ(flag_mu_.size(), engine_thread_num_);
  CHECK_EQ(is_push_finished_.size(), engine_thread_num_);
  CHECK_EQ(q_pull_reqmeta_.size(), engine_thread_num_);
  CHECK_EQ(pull_cnt_.size(), engine_thread_num_);

  // init the engine
  for (size_t i = 0; i < engine_thread_num_; ++i) {
    acc_load_.push_back(0);
  }
  if (sync_mode_) {
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto q = new PriorityQueue(enable_schedule_);
      engine_queues_.push_back(q);
    }
    for (size_t i = 0; i < engine_thread_num_; ++i) {
      auto t = new std::thread(&BytePSServerEngineThread, i);
      engine_threads_.push_back(t);
    }
  }

  // init server instance
  ps::StartPS(0, role_, preferred_rank, true, "byteps\0");
  byteps_server_ = new KVServer<SERVER_DATA_TYPE>(0, false, 0);
  byteps_server_->set_request_handle(BytePSHandler);
  if (!Postoffice::Get()->is_recovery()) {
    Postoffice::Get()->Barrier(
        0, ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
  }

  // clean the server resource
  Finalize(0, role_, true);
  if (byteps_server_) {
    delete byteps_server_;
    byteps_server_ = nullptr;
  }
  if (bps_reducer_) {
    delete bps_reducer_;
    bps_reducer_ = nullptr;
  }
  BytePSEngineMessage msg;
  msg.ops = TERMINATE;
  for (auto q : engine_queues_) q->Push(msg);
  for (auto t : engine_threads_) t->join();

  for (auto& it : store_) {
    if (it.second.tensor) {
      free(it.second.tensor);
    }
  }

  for (auto& it : DGC_store_) {
    if (it.second.tensor) {
      free(it.second.tensor);
    }
  }
  
  LOG(INFO) << "byteps has been shutdown";
#ifdef PS_OVERHEAD
  printf("total PS time: %.6f ms, PS times points: %d\n", total_PS_time, total_PS_time_points);
  printf("total compress time: %.6f ms\n", total_compress_time);
#endif
  return;
}

}  // namespace server
}  // namespace byteps
