#pragma once

#include <cstddef>
#include <memory>
#include <numeric>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/allocator.hpp"

namespace bipp {

  template <std::size_t DIM,std::size_t N>
  struct memory_index_helper_struct{
    inline static auto eval(const std::array<std::size_t, DIM>& indices, const std::array<std::size_t, DIM>& shape,
                            const std::array<std::size_t, DIM>& strides) -> std::size_t {
      return indices[N] * strides[N] +
             memory_index_helper_struct<DIM, N - 1>::eval(indices, shape, strides);
    }
  };

  template <std::size_t DIM>
  struct memory_index_helper_struct<DIM,0>{
    inline static auto eval(const std::array<std::size_t, DIM>& indices, const std::array<std::size_t, DIM>& ,
                            const std::array<std::size_t, DIM>& ) -> std::size_t {
      return indices[0];
    }
  };

template <typename T, std::size_t DIM>
class ConstView {
public:
  using index_type = std::array<std::size_t, DIM>;

  ConstView() {
    shape_.fill(0);
    strides_.fill(0);
  }

  ConstView(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : shape_(shape),
        totalSize_(std::accumulate(shape_.begin(), shape_.end(), std::size_t(0))) {
    if (totalSize_) {
      auto ptr = alloc->allocate(totalSize_ * sizeof(T));
      data_ = std::shared_ptr<void>(ptr, [alloc = std::move(alloc)](void* p) {
        if (p) alloc->deallocate(p);
      });
    }

    strides_[0] = 1;
    for(std::size_t i = 1; i < DIM; ++i) {
      strides_[i] = shape_[i-1] * strides_[i - 1];
    }
  }

  ConstView(const T* ptr, const index_type& shape, const index_type& strides)
      : shape_(shape),
        strides_(strides),
        totalSize_(std::accumulate(shape_.begin(), shape_.end(), std::size_t(0))) {


    if(strides_[0] != 1) throw InvalidStrideError();
    for(std::size_t i = 1; i < DIM; ++i) {
      if (strides_[i] < shape_[i - 1] * strides_[i - 1]) throw InvalidStrideError();
    }
  }

  ConstView(const ConstView&) = default;

  ConstView(ConstView&& b) = default;

  virtual ~ConstView() = default;

  auto operator=(const ConstView&) -> ConstView& = default;

  auto operator=(ConstView&& b) -> ConstView& = default;

  inline auto operator()(const index_type& index) const -> const T& {
    return constPtr_[memory_index(index)];
  }

  inline auto get() const -> const T* { return constPtr_; }

  inline auto shape() const noexcept -> const std::array<std::size_t, DIM> { return shape_; }

  inline auto strides() const noexcept -> const std::array<std::size_t, DIM> { return strides_; }

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

  inline auto has_ownership() const noexcept -> bool { return bool(data_); }

protected:
  inline auto memory_index(const index_type& indices) const -> std::size_t {
    // return memory_index_helper<DIM-1>(indices);
    return memory_index_helper_struct<DIM, DIM-1>::eval(indices, shape_, strides_);
  }

  template <std::size_t N>
  inline auto memory_index_helper(const index_type& indices) const -> std::size_t {
    if constexpr (N == 0) {
      return indices[0];
    } else {
      return indices[N] * strides_[N] + memory_index_helper<N - 1>(indices);
    }
  }



  index_type shape_;
  index_type strides_;
  std::size_t totalSize_;
  std::shared_ptr<void> data_;
  const T* constPtr_ = nullptr;;
};


template <typename T, std::size_t DIM>
class View : public ConstView<T, DIM> {
public:
  using index_type = typename ConstView<T, DIM>::index_type;

  inline auto operator()(const index_type& index) const -> const T& {
    return this->constPtr_[this->memory_index(index)];
  }

  inline auto operator()(const index_type& index) -> T& {
    return const_cast<T*>(this->constPtr_)[this->memory_index(index)];
  }
};


void test() {
  const View<double, 3> v;

  ConstView<double, 3> cv(v);

  auto a = v({0, 0, 0});
  auto b = cv({0, 0, 0});

};



}  // namespace bipp
