/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdlib>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/storage_allocator.h"

namespace tensorflow {

namespace pavo {

StorageAllocator::StorageAllocator() {
}

StorageAllocator::~StorageAllocator() {
}

static void *AllocateRawLocal(size_t alignment, size_t num_bytes) {
  assert((alignment % sizeof(void *)) == 0);
  assert((alignment & (alignment - 1)) == 0);

  void *result;
  int code = posix_memalign(&result, alignment, num_bytes);
  return result;
};

void *StorageAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  LOG(INFO) << "tensorflow::pavo: AllocateRaw no attr " << alignment << " " << num_bytes;

  return AllocateRawLocal(alignment, num_bytes);
}

void *StorageAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                                    const AllocationAttributes &allocation_attr) {
  LOG(INFO) << "tensorflow::pavo: AllocateRaw wt attr " << alignment << " " << num_bytes;
  return AllocateRawLocal(alignment, num_bytes);
}

void StorageAllocator::DeallocateRaw(void *ptr) {
  LOG(INFO) << "tensorflow::pavo: DeallocateRaw";
  free(ptr);
}

bool StorageAllocator::AllocatesOpaqueHandle() const {
  return false;
}

} //  namespace pavo

}  // namespace tensorflow
