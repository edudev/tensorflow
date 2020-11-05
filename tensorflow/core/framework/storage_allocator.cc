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

#undef NDEBUG
#include <cstdlib>
#include <cstdio>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <boost/stacktrace.hpp>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/storage_allocator.h"

namespace tensorflow {

namespace pavo {

const char *MEMORY_DIR = "./pavo/tmp-storage/";
#define FN_BUFFER_SIZE 255

StorageAllocator::StorageAllocator() {
  this->next_block_index_ = 0;
}

StorageAllocator::~StorageAllocator() {
}

size_t StorageAllocator::GetNextBlockIndex() {
  this->memory_map_guard_.lock();
  size_t result = this->next_block_index_;
  ++this->next_block_index_;
  this->memory_map_guard_.unlock();
  return result;
}

static void CreateDirectory() {
  struct stat st;
  if (stat(MEMORY_DIR, &st) == -1) {
    mkdir(MEMORY_DIR, 0700);
  }
}

static int CreateSparseFile(size_t alignment, size_t num_byes, size_t next_fn_num) {
  char fn[FN_BUFFER_SIZE];

  int fd, ret;

  assert((getpagesize() % alignment) == 0);
  CreateDirectory();

  // maybe assert the total length is below FN_BFFER_SIZE
  snprintf(fn, FN_BUFFER_SIZE, "%s%07d.tmp", MEMORY_DIR, next_fn_num);
  ++next_fn_num;

  fd = open(fn, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd < 0) {
    perror("open");
    return -1;
  }

  ret = ftruncate(fd, num_byes);
  if (ret < 0) {
    perror("ftruncate");
    return -1;
  }

  return fd;
}

static void *MemMapFile(size_t alignment, size_t num_bytes, size_t next_fn_num) {
  int fd, ret;

  fd = CreateSparseFile(alignment, num_bytes, next_fn_num);

  if (fd < 0) {
    return NULL;
  }

  void *result = mmap(NULL, num_bytes, PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (result == MAP_FAILED) {
    perror("mmap");
    result = NULL;
  }

  ret = close(fd);
  if (ret < 0) {
    perror("close");
  }

  return result;
}


static void *AllocateRawLocal(size_t alignment, size_t num_bytes, size_t next_fn_num) {
  assert((alignment % sizeof(void *)) == 0);
  assert((alignment & (alignment - 1)) == 0);

  void *result;

//  int code = posix_memalign(&result, alignment, num_bytes);
  result = MemMapFile(alignment, num_bytes, next_fn_num);
  return result;
};

void *StorageAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
  LOG(INFO) << "tensorflow::pavo: AllocateRaw no attr " << alignment << " " << num_bytes;
  std::cerr << boost::stacktrace::stacktrace();

  void *result = AllocateRawLocal(alignment, num_bytes, this->GetNextBlockIndex());

  std::map<const void *, size_t>::const_iterator mmap_it = this->memory_map_.find(result);
  assert(mmap_it == this->memory_map_.cend());

  this->memory_map_guard_.lock();
  this->memory_map_[result] = num_bytes;
  this->memory_map_guard_.unlock();

  return result;
}

static inline bool ends_with(std::string const &value, std::string const &ending) {
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void *StorageAllocator::AllocateRaw(size_t alignment, size_t num_bytes,
                                    const AllocationAttributes &allocation_attr) {
//  auto st = boost::stacktrace::stacktrace();

  std::string alloc_type = "other";
  std::string op_name = "";

//  int i = 0;
//  for (auto it = st.begin(); it != st.end(); ++it) {
//    std::string frame_name = it->name();
//    if (frame_name.find("allocate_output") != std::string::npos) {
//      alloc_type = "allocate_output";
//    } else if (frame_name.find("allocate_temp") != std::string::npos) {
//      alloc_type = "allocate_temp";
//    } else if (frame_name.find("allocate_persistent") != std::string::npos) {
//      alloc_type = "allocate_persistent";
//    }
//
//    if (frame_name.find("tensorflow::") == 0) {
//      // 12 == strlen("tensorflow::")
//      int j = 12;
//      for (; j < frame_name.size(); ++j) {
//        char ch = frame_name.at(j);
//        if (ch >= 'A' && ch <= 'Z') {
//          continue;
//        }
//        if (ch >= 'a' && ch <= 'z') {
//          continue;
//        }
//        break;
//      }
//
//      std::string fn_name = frame_name.substr(12, j - 12);
//
//      if (ends_with(fn_name, "Op")) {
//        if (op_name.size() == 0) {
//          op_name = fn_name;
//        } else {
//          op_name += " & " + fn_name;
//        }
//      }
//    }
//    ++i;
//  }


  void *result = AllocateRawLocal(alignment, num_bytes, this->GetNextBlockIndex());
  LOG(INFO) << "tensorflow::pavo: AllocateRaw wt attr " << alloc_type << " " << op_name << " " << alignment << " "
            << result << " " << num_bytes;

  std::map<const void *, size_t>::const_iterator mmap_it = this->memory_map_.find(result);
  assert(mmap_it == this->memory_map_.cend());

  this->memory_map_guard_.lock();
  this->memory_map_[result] = num_bytes;
  this->memory_map_guard_.unlock();

  return result;
}

void StorageAllocator::DeallocateRaw(void *ptr) {
  this->memory_map_guard_.lock();
  std::map<const void *, size_t>::const_iterator mmap_it = this->memory_map_.find(ptr);
  assert(mmap_it != this->memory_map_.cend());

  size_t num_bytes = mmap_it->second;
  LOG(INFO) << "tensorflow::pavo: DeallocateRaw " << ptr << " " << num_bytes;
  // TODO: figure out why I can't use the iterator here
  this->memory_map_.erase(mmap_it);
  mmap_it = this->memory_map_.find(ptr);
  assert(mmap_it == this->memory_map_.cend());
  this->memory_map_guard_.unlock();
//  this->memory_map_.erase(ptr);
//  free(ptr);

  int ret = munmap(ptr, num_bytes);
  if (ret < 0) {
    perror("munmap");
  }
}

bool StorageAllocator::AllocatesOpaqueHandle() const {
  return false;
}

} //  namespace pavo

}  // namespace tensorflow
