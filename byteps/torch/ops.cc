// Copyright 2019 Bytedance Inc. All Rights Reserved.
// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#include <torch/extension.h>
#include <torch/torch.h>
#include <chrono>
#include <memory>
#include <thread>

#include "../common/global.h"
#include "../common/operations.h"
#include "adapter.h"
#include "ops.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "ready_event.h"

namespace byteps {
namespace torch {

static HandleManager handle_manager;

namespace {

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

int GetDeviceID(const ::torch::Tensor& tensor) {
  if (tensor.device().is_cuda()) {
    return tensor.device().index();
  }
  return CPU_DEVICE_ID;
}

}  // namespace

void InitDeclaredTensor(const std::string& name, int size) {
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  common::InitTensor(context, sizeof(float) * size, BYTEPS_FLOAT32, nullptr);
}

float SendNorm(const std::string& name, float max_norm){
#ifdef DEFAULT_PUSHPULL_INFO
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  // printf("SendNorm: %s, %.6f\n", tensor_name.c_str(), max_norm);
  auto& context = common::GetContextFromName(tensor_name);
  int num_worker = BytePSGlobal::GetNumWorker();
  float norms[num_worker] = {0.0};
  norms[BytePSGlobal::GetWorkerID()] = max_norm;
  // norm_table[context.declared_key] = max_norm;
  common::InitTensor(context, sizeof(float) * num_worker, BYTEPS_FLOAT32, nullptr);
  // common::InitTensor(context, sizeof(float) * num_worker, BYTEPS_FLOAT32, (void*)(&(norm_table[context.declared_key])));
  // common::InitTensor(context, sizeof(float), BYTEPS_FLOAT32, (void*)(&(norm_table[context.declared_key])));
  if(context.initialized){
    auto ps = BytePSGlobal::GetOrInitPS();
    auto &kv = BytePSGlobal::EncodeDefaultKey(context.key_list[0], sizeof(float) * num_worker);
    auto data = reinterpret_cast<char *>(norms);
    ps::SArray<char> vals(data, sizeof(float) * num_worker, false);
    int cmd = GetCommandType(RequestType::kDefaultPushPull, BYTEPS_FLOAT32);
    ps->Wait(ps->ZPush(kv.keys, vals, kv.lens, cmd));

    // false means not to delete data when SArray is deleted
    auto pull_vals = new ps::SArray<char>(data, sizeof(float) * num_worker, false);
    ps->Wait(ps->ZPull(kv.keys, pull_vals, &kv.lens, cmd, [pull_vals]() {
                                   delete pull_vals;
                                 }));
  }
  else{
    printf("Not sending norm becasue the context is uninitialized\n");
  }
  norm_table[context.declared_key] = *(std::max_element(norms, norms+num_worker)); // testing only!!! remove!!!
  // printf("RecvNorm: %s, %.6f\n", tensor_name.c_str(), norm_table[context.declared_key]);
  return norm_table[context.declared_key];
#else
  return max_norm;
#endif // DEFAULT_PUSHPULL_INFO
}

float GetNorm(const std::string& name, int rank){
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  if (context.initialized){
    // printf("context norm for %s In GetNorm: %.6f\n", tensor_name.c_str(), context.norm);
    return context.max_norms[rank];
  }
  else return 0.0;
  
}

int GetNumWorker(){
  return BytePSGlobal::GetNumWorker();
}

int GetWorkerID(){
  return BytePSGlobal::GetWorkerID();
}

int GetLocalNcclRank(){
  // NcclManager::GetRank's arguments are not used in the function
  return BytePSGlobal::GetNccl()->GetRank(0, COPYD2H);
}

int GetLocalSize(){
  return BytePSGlobal::GetLocalSize();
}

size_t GetPartitionBound(){
  return BytePSGlobal::GetPartitionBound();
}

void StartTask(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string tensor_name, int version, int priority, int handle,
               float norm, const std::string compressor_name) {

  // printf("StartTask for tensor %s\n", tensor_name.c_str());
  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto byteps_input = std::make_shared<TorchTensor>(tensor);
  auto byteps_output = std::make_shared<TorchTensor>(output);
  size_t size = byteps_input->size();
  auto dtype = byteps_input->dtype();

  auto& context = common::GetContextFromName(tensor_name);
  context.compressor_name = compressor_name;
  if (size == 0) {
    // Skip empty tensors.
    printf("Skip empty tensor %s\n", tensor_name.c_str());
    handle_manager.MarkDone(handle, Status::OK());
    return;
  }
  common::InitTensor(context, size, dtype,
                      (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr);

  // the GetPushQueueList returns a list containing queues for all
  // operations before pushing (e.g., REDUCE, COPYD2H) and PUSH
  // the GetPullQueueList returns a list containing PULL and queues for all operations
  // after pulling (e.g., COPYH2D, BROADCAST). See operations.c line 455-511
  std::shared_ptr<std::vector<QueueType>> queue_list;
  if(tensor_name.find("Uncompress") != std::string::npos) {
    if (compressor_name == "inca"){
      queue_list = common::GetContextQueueList(device);
    }
    else{
      queue_list = common::GetPushQueueList(device);
      auto queue_list_pull = common::GetPullQueueList(device);
      queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());
    }
  }
  else{
    if(tensor_name.find("Compress") != std::string::npos) {
      queue_list = common::GetAfterContextPushQueueList(device);
    }
    else {
      queue_list = common::GetPushQueueList(device);
    }
    auto queue_list_pull = common::GetPullQueueList(device);
    queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());
  }

  auto enqueue_result = common::EnqueueTensor(
      context, byteps_input, byteps_output, ready_event, device, priority,
      version,
      [handle, average, tensor, output](const Status& status) mutable {
        // Will execute in the `device` context.
        if (average) {
#if TORCH_VERSION >= 1005000000
          if (isIntegralType(output.scalar_type())) {
            output.floor_divide_(byteps_size());
            handle_manager.MarkDone(handle, status);
            return;
          }
#endif
          output.div_(byteps_size());
        }
        handle_manager.MarkDone(handle, status);
      },
      queue_list,
      norm);

  ThrowIfError(enqueue_result);
  return;

}

int DoPushPull(::torch::Tensor tensor, ::torch::Tensor output, int average,
               const std::string& name, int version, int priority, float norm=0.0,
               const std::string& compressor_name="") {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  // context.actual_len = actual_len;
  if (context.initialized) {
    StartTask(tensor, output, average, tensor_name, version, priority, handle, norm, compressor_name);
  } else {
    std::thread t(StartTask, tensor, output, average, tensor_name, version, priority, handle, norm, compressor_name);
    t.detach();
  }
  return handle;
}

void SetNumGrads(int num_grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  num_grads_ = num_grads;
  grad_count_ = 0;
  return;
}

int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void DeclareTensor(const std::string& name) {
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  common::IsTensorDeclared(tensor_name);
}

void WaitAndClear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

pybind11::tuple DoPushPullGroupSync(::torch::Tensor tensor,
                                    ::torch::Tensor output, int average,
                                    const std::string& name, int version,
                                    int priority, float norm=0.0, const std::string& compressor_name="") {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  std::string tensor_name = GetOpName("byteps", name.c_str(), 0);
  auto& context = common::GetContextFromName(tensor_name);
  int curr_count;

  if (context.initialized) {
    StartTask(tensor, output, average, tensor_name, version, priority, handle, norm, compressor_name);
  } else {
    std::thread t(StartTask, tensor, output, average, tensor_name, version,
                  priority, handle, norm, compressor_name);
    t.detach();
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    grad_count_++;
    curr_count = grad_count_;
    if (grad_count_ == num_grads_) {
      grad_count_ = 0;
    }
  }

  return pybind11::make_tuple(handle, curr_count);
}

PYBIND11_MODULE(c_lib, m) {
  // push_pull
  m.def("byteps_torch_push_pull_async_torch_ByteTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_IntTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_LongTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_HalfTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_FloatTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_DoubleTensor", &DoPushPull);

  m.def("byteps_torch_set_num_grads", &SetNumGrads);

  m.def("byteps_torch_push_pull_group_sync_torch_ByteTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_IntTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_LongTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_HalfTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_FloatTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_DoubleTensor", &DoPushPullGroupSync);

#if HAVE_CUDA
  m.def("byteps_torch_push_pull_async_torch_cuda_ByteTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_IntTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_LongTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_HalfTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_FloatTensor", &DoPushPull);
  m.def("byteps_torch_push_pull_async_torch_cuda_DoubleTensor", &DoPushPull);

  m.def("byteps_torch_push_pull_group_sync_torch_cuda_ByteTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_IntTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_LongTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_HalfTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_FloatTensor", &DoPushPullGroupSync);
  m.def("byteps_torch_push_pull_group_sync_torch_cuda_DoubleTensor", &DoPushPullGroupSync);
#endif

  // basics
  m.def("byteps_torch_poll", &PollHandle);
  m.def("byteps_torch_wait_and_clear", &WaitAndClear);
  m.def("byteps_torch_declare_tensor", &DeclareTensor);
  m.def("byteps_torch_init_declared_tensor", &InitDeclaredTensor);

  m.def("byteps_torch_get_num_worker", &GetNumWorker);
  m.def("byteps_torch_get_worker_id", &GetWorkerID);
  m.def("byteps_torch_get_local_rank", &GetLocalNcclRank);
  m.def("byteps_torch_get_local_size", &GetLocalSize);
  m.def("byteps_torch_get_partition_bound", &GetPartitionBound);
  // communicate the norm information
  m.def("byteps_torch_send_norm", &SendNorm);
  m.def("byteps_torch_get_norm", &GetNorm); 
}

}  // namespace torch
}  // namespace byteps
