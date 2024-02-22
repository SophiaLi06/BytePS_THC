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

#include "ready_table.h"

#include "logging.h"

namespace byteps {
namespace common {

// below are methods for accessing/modifying the _ready_table
bool ReadyTable::IsKeyReady(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  return _ready_table[key] == (_ready_count);
}

int ReadyTable::AddReadyCount(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  BPS_CHECK_LT(_ready_table[key], _ready_count)
      << _table_name << ": " << _ready_table[key] << ", " << (_ready_count);
  return ++_ready_table[key];
}

int ReadyTable::SetReadyCount(uint64_t key, int cnt) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  _ready_table[key] = cnt;
}

void ReadyTable::ClearReadyCount(uint64_t key) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  _ready_table[key] = 0;
}

float ReadyTable::GetKeyScale(uint64_t key, int rank) {
  std::lock_guard<std::mutex> lock(_table_mutex);
  return _scale_table[key][rank];
}

void ReadyTable::SetKeyScale(uint64_t key, float scale, int rank){
  std::lock_guard<std::mutex> lock(_table_mutex);
  _scale_table[key][rank] = scale;
  // if (scale > _scale_table[key]) _scale_table[key] = scale;
}

float ReadyTable::GetKeyNorm(uint64_t key, int rank){
  std::lock_guard<std::mutex> lock(_table_mutex);
  return _norm_table[key][rank];
}

void ReadyTable::SetKeyNorm(uint64_t key, float max_norm, int rank){
  std::lock_guard<std::mutex> lock(_table_mutex);
  _norm_table[key][rank] = max_norm;
  // if (max_norm > _norm_table[key]) _norm_table[key] = max_norm;
}

}  // namespace common
}  // namespace byteps
