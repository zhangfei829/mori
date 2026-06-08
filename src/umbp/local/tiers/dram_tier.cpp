// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // for sched_getaffinity / sched_setaffinity / CPU_* macros
#endif

#include "umbp/local/tiers/dram_tier.h"

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace mori::umbp {

namespace {

#if defined(__x86_64__) || defined(__i386__)
// Non-temporal (streaming) copy: writes bypass the cache, skipping the
// read-for-ownership of destination lines that a normal store incurs. Ideal
// when the destination (L2 staging buffer) is written once and then DMA'd to
// HBM without being re-read by the CPU. Handles unaligned head/tail.
__attribute__((target("avx2"))) void NtCopy(char* d, const char* s, size_t n) {
  size_t head = (32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31;
  if (head > n) head = n;
  std::memcpy(d, s, head);
  size_t i = head;
  for (; i + 128 <= n; i += 128) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 32));
    __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 64));
    __m256i e = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + 96));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 32), b);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 64), c);
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i + 96), e);
  }
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
    _mm256_stream_si256(reinterpret_cast<__m256i*>(d + i), a);
  }
  if (i < n) std::memcpy(d + i, s + i, n - i);
  _mm_sfence();
}
bool NtSupported() { return __builtin_cpu_supports("avx2"); }
#else
void NtCopy(char* d, const char* s, size_t n) { std::memcpy(d, s, n); }
bool NtSupported() { return false; }
#endif

// Parse a Linux cpulist string like "0-3,8,12-15" into individual cpu ids.
std::vector<int> ParseCpuList(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) continue;
    auto dash = tok.find('-');
    if (dash == std::string::npos) {
      out.push_back(std::atoi(tok.c_str()));
    } else {
      int a = std::atoi(tok.substr(0, dash).c_str());
      int b = std::atoi(tok.substr(dash + 1).c_str());
      for (int i = a; i <= b; ++i) out.push_back(i);
    }
  }
  return out;
}

// SMT siblings of |cpu| (incl. itself) from sysfs; falls back to {cpu}.
std::vector<int> ReadThreadSiblings(int cpu) {
  std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                  "/topology/thread_siblings_list");
  if (!f) return {cpu};
  std::string line;
  std::getline(f, line);
  auto v = ParseCpuList(line);
  return v.empty() ? std::vector<int>{cpu} : v;
}

}  // namespace

DRAMTier::DRAMTier(size_t capacity, bool use_shm, const std::string& shm_name, bool use_hugepages,
                   size_t hugepage_size, int numa_node, bool prefault)
    : TierBackend(StorageTier::CPU_DRAM),
      base_ptr_(nullptr),
      capacity_(capacity),
      mapped_size_(0),
      used_(0),
      shm_fd_(-1),
      use_shm_(use_shm),
      shm_name_(shm_name) {
  if (use_shm_) {
    shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ < 0) {
      throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
    }
    if (ftruncate(shm_fd_, capacity_) < 0) {
      close(shm_fd_);
      shm_unlink(shm_name_.c_str());
      throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
    }
    base_ptr_ = mmap(nullptr, capacity_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (base_ptr_ == MAP_FAILED) {
      close(shm_fd_);
      shm_unlink(shm_name_.c_str());
      throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
    }
    mapped_size_ = capacity_;
  } else {
    HostMemAllocator allocator;
    HostBufferOptions opts;
    opts.backing =
        use_hugepages ? HostBufferBacking::kAnonymousHugetlb : HostBufferBacking::kAnonymous;
    opts.hugepage_size = hugepage_size;
    opts.numa_node = numa_node;
    opts.prefault = prefault;

    host_buf_handle_ = allocator.Alloc(capacity_, opts);
    if (!host_buf_handle_.valid()) {
      throw std::runtime_error("DRAMTier: memory allocation failed for " +
                               std::to_string(capacity_) + " bytes");
    }
    base_ptr_ = host_buf_handle_.ptr;
    mapped_size_ = host_buf_handle_.mapped_size;
  }

  // Initialize free list with entire capacity
  free_list_.push_back({0, capacity_});

  // Parallel-memcpy fan-out for batch reads. Default 4 (with non-temporal
  // stores that is ~76 GiB/s on EPYC node-local DRAM, comfortably above the
  // downstream HBM-DMA demand), overridable via env, capped to hw threads.
  read_threads_ = 4;
  if (const char* env = std::getenv("UMBP_DRAM_READ_THREADS")) {
    long v = std::atol(env);
    if (v > 0) read_threads_ = static_cast<size_t>(v);
  }
  unsigned hw = std::thread::hardware_concurrency();
  if (hw > 0 && read_threads_ > hw) read_threads_ = hw;

  // Worker CPU pinning (default on). Respect the operator's core budget by
  // reading our own affinity mask (taskset / cgroup) rather than assuming the
  // whole machine.
  pin_threads_ = true;
  if (const char* env = std::getenv("UMBP_DRAM_READ_PIN")) {
    pin_threads_ = !(env[0] == '0');
  }
  nt_copy_ = NtSupported();
  if (const char* env = std::getenv("UMBP_DRAM_NT_COPY")) {
    nt_copy_ = nt_copy_ && !(env[0] == '0');
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int c = 0; c < CPU_SETSIZE; ++c) {
      if (CPU_ISSET(c, &set)) allowed_cpus_.push_back(c);
    }
  }

  // De-duplicate to one representative (lowest sibling) per physical core, so
  // pinning never lands two memcpy threads on SMT siblings of the same core.
  std::set<int> allowed_set(allowed_cpus_.begin(), allowed_cpus_.end());
  std::set<int> covered;
  for (int c : allowed_cpus_) {
    if (covered.count(c)) continue;
    auto sibs = ReadThreadSiblings(c);
    int rep = c;
    for (int s : sibs) rep = std::min(rep, s);
    for (int s : sibs) covered.insert(s);
    if (allowed_set.count(rep)) allowed_phys_cpus_.push_back(rep);
  }
  if (allowed_phys_cpus_.empty()) allowed_phys_cpus_ = allowed_cpus_;
  std::sort(allowed_phys_cpus_.begin(), allowed_phys_cpus_.end());

  // Never spawn more memcpy threads than the physical cores we're allowed to
  // use, so batch reads don't oversubscribe the inference engine's CPU budget.
  if (pin_threads_ && !allowed_phys_cpus_.empty() &&
      read_threads_ > allowed_phys_cpus_.size()) {
    read_threads_ = allowed_phys_cpus_.size();
  }
}

DRAMTier::~DRAMTier() {
  if (use_shm_) {
    if (base_ptr_ && base_ptr_ != MAP_FAILED) {
      munmap(base_ptr_, mapped_size_);
    }
    if (shm_fd_ >= 0) close(shm_fd_);
    shm_unlink(shm_name_.c_str());
  } else {
    HostMemAllocator allocator;
    allocator.Free(host_buf_handle_);
  }
}

size_t DRAMTier::Allocate(size_t size) {
  // First-fit allocation
  for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
    if (it->size >= size) {
      size_t offset = it->offset;
      if (it->size == size) {
        free_list_.erase(it);
      } else {
        it->offset += size;
        it->size -= size;
      }
      return offset;
    }
  }
  return static_cast<size_t>(-1);  // Allocation failed
}

void DRAMTier::Deallocate(size_t offset, size_t size) {
  // Insert into sorted position and coalesce adjacent blocks
  auto it = free_list_.begin();
  while (it != free_list_.end() && it->offset < offset) {
    ++it;
  }

  auto new_it = free_list_.insert(it, {offset, size});

  // Coalesce with next block
  auto next = std::next(new_it);
  if (next != free_list_.end() && new_it->offset + new_it->size == next->offset) {
    new_it->size += next->size;
    free_list_.erase(next);
  }

  // Coalesce with previous block
  if (new_it != free_list_.begin()) {
    auto prev = std::prev(new_it);
    if (prev->offset + prev->size == new_it->offset) {
      prev->size += new_it->size;
      free_list_.erase(new_it);
    }
  }
}

void DRAMTier::TouchLRU(const std::string& key) {
  auto it = lru_map_.find(key);
  if (it != lru_map_.end()) {
    lru_list_.erase(it->second);
  }
  lru_list_.push_front(key);
  lru_map_[key] = lru_list_.begin();
}

void DRAMTier::EvictLRU() {
  // Caller must hold data_mu_ (unique). LRU structures guarded by lru_mu_.
  std::lock_guard<std::mutex> llock(lru_mu_);
  if (lru_list_.empty()) return;

  const std::string victim = lru_list_.back();
  auto slot_it = slots_.find(victim);
  if (slot_it != slots_.end()) {
    Deallocate(slot_it->second.offset, slot_it->second.size);
    used_ -= slot_it->second.size;
    slots_.erase(slot_it);
  }
  lru_map_.erase(victim);
  lru_list_.pop_back();
}

bool DRAMTier::Write(const std::string& key, const void* data, size_t size) {
  std::unique_lock<std::shared_mutex> lock(data_mu_);

  // If key already exists, free its old slot first
  auto existing = slots_.find(key);
  if (existing != slots_.end()) {
    Deallocate(existing->second.offset, existing->second.size);
    used_ -= existing->second.size;
    slots_.erase(existing);
    std::lock_guard<std::mutex> llock(lru_mu_);
    auto lru_it = lru_map_.find(key);
    if (lru_it != lru_map_.end()) {
      lru_list_.erase(lru_it->second);
      lru_map_.erase(lru_it);
    }
  }

  // Try to allocate — do NOT self-evict.
  // If no space, return false so upper layer can demote keys to SSD.
  size_t offset = Allocate(size);
  if (offset == static_cast<size_t>(-1)) {
    return false;
  }

  std::memcpy(static_cast<char*>(base_ptr_) + offset, data, size);
  slots_[key] = {offset, size};
  used_ += size;
  {
    std::lock_guard<std::mutex> llock(lru_mu_);
    TouchLRU(key);
  }
  return true;
}

bool DRAMTier::ReadIntoPtr(const std::string& key, uintptr_t dst_ptr, size_t size) {
  // Shared lock: concurrent reads proceed in parallel. The unique-locking
  // Write/Evict/Clear paths are excluded for the duration, so the resolved
  // slot memory is stable across the memcpy.
  std::shared_lock<std::shared_mutex> lock(data_mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  // Reject if caller's buffer size does not match the stored block size.
  // A mismatch indicates a caller bug (wrong page size); silently truncating
  // would produce a partially-filled KV block with no error signal.
  if (size != it->second.size) return false;

  std::memcpy(reinterpret_cast<void*>(dst_ptr), static_cast<char*>(base_ptr_) + it->second.offset,
              size);
  {
    std::lock_guard<std::mutex> llock(lru_mu_);
    TouchLRU(key);
  }
  return true;
}

const void* DRAMTier::ReadPtr(const std::string& key, size_t* out_size) {
  std::shared_lock<std::shared_mutex> lock(data_mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return nullptr;

  if (out_size) *out_size = it->second.size;
  {
    std::lock_guard<std::mutex> llock(lru_mu_);
    TouchLRU(key);
  }
  return static_cast<char*>(base_ptr_) + it->second.offset;
}

std::vector<char> DRAMTier::Read(const std::string& key) {
  std::shared_lock<std::shared_mutex> lock(data_mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return {};

  size_t sz = it->second.size;
  std::vector<char> buf(sz);
  std::memcpy(buf.data(), static_cast<char*>(base_ptr_) + it->second.offset, sz);
  {
    std::lock_guard<std::mutex> llock(lru_mu_);
    TouchLRU(key);
  }
  return buf;
}

std::vector<bool> DRAMTier::ReadBatchIntoPtr(const std::vector<std::string>& keys,
                                             const std::vector<uintptr_t>& dst_ptrs,
                                             const std::vector<size_t>& sizes) {
  const size_t n = keys.size();
  std::vector<bool> results(n, false);
  if (n == 0) return results;

  std::vector<const char*> srcs(n, nullptr);
  std::vector<size_t> hits;
  hits.reserve(n);

  // Hold the shared lock for the entire operation (including the parallel
  // memcpy below). Workers are joined before the lock is released, so no
  // unique-locking writer/evictor can move or reuse a slot mid-copy.
  std::shared_lock<std::shared_mutex> lock(data_mu_);

  for (size_t i = 0; i < n; ++i) {
    auto it = slots_.find(keys[i]);
    if (it == slots_.end()) continue;
    if (sizes[i] != it->second.size) continue;  // size mismatch => caller bug
    srcs[i] = static_cast<const char*>(base_ptr_) + it->second.offset;
    hits.push_back(i);
  }

  const size_t num_hits = hits.size();
  if (num_hits == 0) return results;

  // |cpu| >= 0 pins the calling thread to that core before copying. Workers
  // are pinned to distinct entries of allowed_cpus_ (low ids first => separate
  // physical cores, avoiding SMT-sibling contention); the main thread runs its
  // chunk unpinned to avoid disturbing the caller's affinity.
  auto do_copy = [&](size_t begin, size_t end, int cpu) {
    if (cpu >= 0) {
      cpu_set_t s;
      CPU_ZERO(&s);
      CPU_SET(cpu, &s);
      sched_setaffinity(0, sizeof(s), &s);
    }
    for (size_t k = begin; k < end; ++k) {
      const size_t i = hits[k];
      char* d = reinterpret_cast<char*>(dst_ptrs[i]);
      if (nt_copy_) {
        NtCopy(d, srcs[i], sizes[i]);
      } else {
        std::memcpy(d, srcs[i], sizes[i]);
      }
    }
  };

  const size_t nthreads = std::min<size_t>(read_threads_, num_hits);
  const bool pin = pin_threads_ && !allowed_phys_cpus_.empty();
  const size_t P = allowed_phys_cpus_.size();
  if (nthreads <= 1) {
    do_copy(0, num_hits, -1);
  } else {
    const size_t chunk = (num_hits + nthreads - 1) / nthreads;
    std::vector<std::thread> workers;
    workers.reserve(nthreads - 1);
    for (size_t t = 1; t < nthreads; ++t) {
      const size_t b = t * chunk;
      if (b >= num_hits) break;
      const size_t e = std::min(num_hits, b + chunk);
      // Stride across physical cores so workers spread over EPYC CCDs (distinct
      // GMI links) rather than piling onto one CCD.
      const int cpu = pin ? allowed_phys_cpus_[(t * P / nthreads) % P] : -1;
      workers.emplace_back(do_copy, b, e, cpu);
    }
    do_copy(0, std::min(num_hits, chunk), -1);  // main thread takes the first chunk
    for (auto& w : workers) w.join();
  }

  {
    std::lock_guard<std::mutex> llock(lru_mu_);
    for (size_t k = 0; k < num_hits; ++k) {
      const size_t i = hits[k];
      results[i] = true;
      TouchLRU(keys[i]);
    }
  }
  return results;
}

TierCapabilities DRAMTier::Capabilities() const {
  TierCapabilities caps;
  caps.zero_copy_read = true;
  caps.batch_read = true;
  return caps;
}

bool DRAMTier::Exists(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(data_mu_);
  return slots_.count(key) > 0;
}

bool DRAMTier::Evict(const std::string& key) {
  std::unique_lock<std::shared_mutex> lock(data_mu_);

  auto it = slots_.find(key);
  if (it == slots_.end()) return false;

  Deallocate(it->second.offset, it->second.size);
  used_ -= it->second.size;
  slots_.erase(it);

  std::lock_guard<std::mutex> llock(lru_mu_);
  auto lru_it = lru_map_.find(key);
  if (lru_it != lru_map_.end()) {
    lru_list_.erase(lru_it->second);
    lru_map_.erase(lru_it);
  }
  return true;
}

std::pair<size_t, size_t> DRAMTier::Capacity() const {
  std::shared_lock<std::shared_mutex> lock(data_mu_);
  return {used_, capacity_};
}

void DRAMTier::Clear() {
  std::unique_lock<std::shared_mutex> lock(data_mu_);
  slots_.clear();
  free_list_.clear();
  free_list_.push_back({0, capacity_});
  used_ = 0;

  std::lock_guard<std::mutex> llock(lru_mu_);
  lru_list_.clear();
  lru_map_.clear();
}

std::vector<std::string> DRAMTier::GetLRUCandidates(size_t max_candidates) const {
  if (max_candidates == 0) max_candidates = 1;
  std::lock_guard<std::mutex> llock(lru_mu_);
  std::vector<std::string> result;
  result.reserve(std::min(max_candidates, lru_list_.size()));
  // Walk from the back (LRU end) up to max_candidates entries.
  auto it = lru_list_.rbegin();
  for (size_t i = 0; i < max_candidates && it != lru_list_.rend(); ++i, ++it) {
    result.push_back(*it);
  }
  return result;
}

std::string DRAMTier::GetLRUKey() const {
  std::lock_guard<std::mutex> llock(lru_mu_);
  if (lru_list_.empty()) return "";
  return lru_list_.back();
}

std::optional<size_t> DRAMTier::GetSlotOffset(const std::string& key) const {
  std::shared_lock<std::shared_mutex> lock(data_mu_);
  auto it = slots_.find(key);
  if (it == slots_.end()) return std::nullopt;
  return it->second.offset;
}

std::optional<std::string> DRAMTier::GetLocationId(const std::string& key) const {
  auto offset = GetSlotOffset(key);
  if (!offset.has_value()) {
    return std::nullopt;
  }
  return std::to_string(*offset);
}

}  // namespace mori::umbp
