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
#include "tensorflow/core/framework/storage_allocator.h"

namespace tensorflow {

namespace pavo {

StorageAllocator::StorageAllocator(Allocator *allocator) :

    allocator_(allocator) {}

StorageAllocator::~StorageAllocator() {}

void *StorageAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  LOG(INFO) << "tensorflow::pavo: AllocateRaw no attr " << alignment << " " << num_bytes;
  return this->allocator_->AllocateRaw(alignment, num_bytes);
}

void *StorageAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                                    const AllocationAttributes &allocation_attr) {
  LOG(INFO) << "tensorflow::pavo: AllocateRaw wt attr " << alignment << " " << num_bytes;
  return this->allocator_->AllocateRaw(alignment, num_bytes, allocation_attr);
}

void StorageAllocator::DeallocateRaw(void *ptr) {
  LOG(INFO) << "tensorflow::pavo: DeallocateRaw";
  this->allocator_->DeallocateRaw(ptr);
}

bool StorageAllocator::TracksAllocationSizes() const {
  return this->allocator_->TracksAllocationSizes();
}

bool StorageAllocator::AllocatesOpaqueHandle() const {
  return this->allocator_->AllocatesOpaqueHandle();
}

size_t StorageAllocator::RequestedSize(const void *ptr) const {
  return this->allocator_->RequestedSize(ptr);
}

size_t StorageAllocator::AllocatedSize(const void *ptr) const {
  return this->allocator_->AllocatedSize(ptr);
}

int64 StorageAllocator::AllocationId(const void *ptr) const {
  return this->allocator_->AllocationId(ptr);
}

size_t StorageAllocator::AllocatedSizeSlow(const void *ptr) const {
  return this->allocator_->AllocatedSizeSlow(ptr);
}

absl::optional<AllocatorStats> StorageAllocator::GetStats() {
  return this->allocator_->GetStats();
}

void StorageAllocator::ClearStats() {
  this->allocator_->ClearStats();
}

void StorageAllocator::SetSafeFrontier(uint64 count) {
  this->allocator_->SetSafeFrontier(count);
}

} //  namespace pavo

}  // namespace tensorflow
