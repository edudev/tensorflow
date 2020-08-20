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

#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {

namespace pavo {

class StorageAllocator : public Allocator {
 public:
  explicit StorageAllocator();
  ~StorageAllocator() override;

  string Name() override { return "storage"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void* AllocateRaw(size_t alignment, size_t num_bytes,
                              const AllocationAttributes& allocation_attr) override;
    void DeallocateRaw(void* ptr) override;
    bool AllocatesOpaqueHandle() const override;
private:
  TF_DISALLOW_COPY_AND_ASSIGN(StorageAllocator);
};

}  // namespace

}  // namespace tensorflow
