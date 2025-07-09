#pragma once
#include <vector>

#include "cuda_runtime.h"

namespace common {
template<typename T> class device_static_vector {
public:
  device_static_vector(): data_(nullptr), size_(0) {
  }
  device_static_vector(const T *host_data, const size_t size): size_(size) {
    cudaMalloc(&data_, size_ * sizeof(T));
    cudaMemcpy(data_, host_data, size_ * sizeof(T), cudaMemcpyHostToDevice);
  }
  explicit device_static_vector(const std::vector<T> &init): device_static_vector(init.data(), init.size()) {
  }
  explicit device_static_vector(size_t size): size_(size) {
    cudaMalloc(&data_, size_ * sizeof(T));
  }

public:
  device_static_vector(device_static_vector &&other) noexcept {
    size_ = other.size_;
    data_ = other.data_;
    other.size_ = 0;
    other.data_ = nullptr;
  }
  device_static_vector &operator=(device_static_vector &&other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      size_ = other.size_;
      data_ = other.data_;
      other.size_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }
  device_static_vector(const device_static_vector &other) {
    size_ = other.size_;
    cudaMalloc(&data_, size_ * sizeof(T));
    cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
  }
  device_static_vector &operator=(const device_static_vector &other) {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      size_ = other.size_;
      cudaMalloc(&data_, size_ * sizeof(T));
      cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    return *this;
  }
  ~device_static_vector() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

public:
  std::vector<T> to_vector() const {
    std::vector<T> res(size_);
    cudaMemcpy(res.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    return res;
  }
  T *data() { return data_; }
  const T *data() const { return data_; }
  [[nodiscard]] size_t size() const { return size_; }

private:
  T *data_;
  size_t size_;
};
}
