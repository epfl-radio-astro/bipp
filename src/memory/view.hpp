#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <type_traits>

#include "bipp/config.h"
#include "bipp/exceptions.hpp"
#include "memory/allocator.hpp"


/*
 *
 *  Views are non-owning objects allowing access to memory through multi-dimensional indexing.
 *  Arrays inherit from views and own the associated memory.
 *
 *  Note: Coloumn-major memory layout! The stride in the first dimension is always 1.
 *
 *  The inheritance tree is as follows:
 *
 *                            ConstView
 *                                |
 *                ------------------------View
 *                |                         |
 *          --------------              -----------
 *          |            |              |         |
 *  ConstHostView  ConstDeviceView  HostView DeviceView
 *                                      |         |
 *                                 HostArray DeviceArray
 *
 *
 * Host views support the [..] operator for element-wise access.
 * Device views may not be passed to device kernels and do not support element-wise access.
 */

namespace bipp {

namespace impl {
// Use specialized structs to compute index, since some compiler do not properly optimize otherwise
template <std::size_t DIM, std::size_t N>
struct ViewIndexHelper {
  inline static constexpr auto eval(const std::array<std::size_t, DIM>& indices,
                                    const std::array<std::size_t, DIM>& strides) -> std::size_t {
    return indices[N] * strides[N] + ViewIndexHelper<DIM, N - 1>::eval(indices, strides);
  }
};

template <std::size_t DIM>
struct ViewIndexHelper<DIM, 0> {
  inline static constexpr auto eval(const std::array<std::size_t, DIM>& indices,
                                    const std::array<std::size_t, DIM>&) -> std::size_t {
    static_assert(DIM >= 1);
    return indices[0];
  }
};

template <std::size_t DIM>
struct ViewIndexHelper<DIM, 1> {
  inline static constexpr auto eval(const std::array<std::size_t, DIM>& indices,
                                    const std::array<std::size_t, DIM>& strides) -> std::size_t {
    static_assert(DIM >= 2);
    return indices[0] + indices[1] * strides[1];
  }
};

}  // namespace impl

/*
 * Helper functions
 */
template <std::size_t DIM>
inline constexpr auto view_index(const std::array<std::size_t, DIM>& indices,
                                  const std::array<std::size_t, DIM>& strides) -> std::size_t {
  return impl::ViewIndexHelper<DIM, DIM - 1>::eval(indices, strides);
}

inline constexpr auto view_index(std::size_t index, std::size_t) -> std::size_t { return index; }

inline constexpr auto view_size(std::size_t shape) -> std::size_t { return shape; }

template <std::size_t DIM>
inline constexpr auto view_size(const std::array<std::size_t, DIM>& shape) -> std::size_t {
  return std::reduce(shape.begin(), shape.end(), std::size_t(1), std::multiplies{});
}



template <typename T, std::size_t DIM>
class ConstView {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = std::conditional_t<DIM == 1, std::size_t, std::array<std::size_t, DIM>>;
  using SliceType = ConstView<T, DIM - 1>;

  ConstView() {
    if constexpr(DIM==1) {
      shape_ = 0;
      strides_ = 1;
    } else {
      shape_.fill(0);
      strides_.fill(1);
    }
  }

  ConstView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : shape_(shape), strides_(strides), totalSize_(view_size(shape)), constPtr_(ptr) {
#ifndef NDEBUG
    assert(this->strides(0) == 1);
    for (std::size_t i = 1; i < DIM; ++i) {
      assert(this->strides(i) >= this->shape(i - 1) * this->strides(i - 1));
    }
#endif
  }

  virtual ~ConstView() = default;

  inline auto data() const -> const T* { return constPtr_; }

  inline auto size() const noexcept -> std::size_t { return totalSize_; }

  inline auto size_in_bytes() const noexcept -> std::size_t { return totalSize_ * sizeof(T); }

  inline auto shape() const noexcept -> const IndexType& { return shape_; }

  inline auto shape(std::size_t i) const noexcept -> std::size_t {
    assert(i < DIM);
    if constexpr (DIM == 1)
      return shape_;
    else
      return shape_[i];
  }

  inline auto strides() const noexcept -> const IndexType& { return strides_; }

  inline auto strides(std::size_t i) const noexcept -> std::size_t {
    assert(i < DIM);
    if constexpr (DIM == 1)
      return strides_;
    else
      return strides_[i];
  }

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> ConstView<T, DIM> {
    return this->template sub_view_impl<ConstView<T, DIM>>(offset, shape);
  }

protected:
  friend ConstView<T, DIM + 1>;

  template <typename UnaryTransformOp>
  inline auto compare_elements(const IndexType& left, const IndexType& right,
                               UnaryTransformOp&& op) const -> bool {
    if constexpr (DIM == 1) {
      return op(left, right);
    } else {
      return std::transform_reduce(left.begin(), left.end(), right.begin(), true,
                                   std::logical_and{}, std::forward<UnaryTransformOp>(op));
    }
  }

  template <typename SLICE_TYPE>
  auto slice_view_impl(std::size_t outer_index) const -> SLICE_TYPE {
    assert(outer_index < this->shape(DIM - 1));

    typename SLICE_TYPE::IndexType sliceShape, sliceStrides;
    if constexpr(DIM == 2) {
      sliceShape = shape_[0];
      sliceStrides = strides_[0];
    } else {
      std::copy(this->shape_.begin(), this->shape_.end() - 1, sliceShape.begin());
      std::copy(this->strides_.begin(), this->strides_.end() - 1, sliceStrides.begin());
    }

    return SLICE_TYPE{ConstView<T, DIM - 1>{this->constPtr_ + outer_index * this->strides(DIM - 1),
                                           sliceShape, sliceStrides}};
  }

  template <typename VIEW_TYPE>
  auto sub_view_impl(const IndexType& offset, const IndexType& shape) const -> VIEW_TYPE {
    assert(compare_elements(offset, shape_, std::less{}));
#ifndef NDEBUG
    for (std::size_t i = 0; i < DIM; ++i) {
      if constexpr (DIM == 1) {
        assert(shape + offset <= shape_);
      } else {
        assert(shape[i] + offset[i] <= shape_[i]);
      }
    }
#endif

    return VIEW_TYPE{ConstView{constPtr_ + view_index(offset, strides_), shape, strides_}};
  }

  IndexType shape_;
  IndexType strides_;
  std::size_t totalSize_ = 0;
  const T* constPtr_ = nullptr;
};

template <typename T, std::size_t DIM>
class View : public ConstView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = View<T, DIM - 1>;

  View() : BaseType() {}

  View(T* ptr, const IndexType& shape, const IndexType& strides) : BaseType(ptr, shape, strides) {}

  inline auto data() -> T* { return const_cast<T*>(this->constPtr_); }

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> View<T, DIM> {
    return this->template sub_view_impl<View<T, DIM>>(offset, shape);
  }

protected:
  friend ConstView<T, DIM>;
  friend ConstView<T, DIM + 1>;

  View(const BaseType& v) : BaseType(v) {}
};

template <typename T, std::size_t DIM>
class HostView : public View<T, DIM> {
public:
  using ValueType = T;
  using BaseType = View<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = HostView<T, DIM - 1>;

  HostView() : BaseType(){};

  explicit HostView(const View<T, DIM>& v) : BaseType(v){};

  HostView(T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  inline auto operator[](const IndexType& index) const -> const T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return this->constPtr_[view_index(index, this->strides_)];
  }

  inline auto operator[](const IndexType& index) -> T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return const_cast<T*>(this->constPtr_)[view_index(index, this->strides_)];
  }

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> HostView<T, DIM> {
    return this->template sub_view_impl<HostView<T, DIM>>(offset, shape);
  }

  auto zero() -> void {
    if (this->totalSize_) {
      if constexpr (DIM <= 1) {
        std::memset(this->data(), 0, this->shape_ * sizeof(T));
      } else {
        for (std::size_t i = 0; i < this->shape_[DIM - 1]; ++i) this->slice_view(i).zero();
      }
    }
  }

protected:
  friend ConstView<T, DIM>;
  friend ConstView<T, DIM + 1>;

  HostView(const ConstView<T, DIM>& b) : BaseType(b){};
};


template <typename T, std::size_t DIM>
class ConstHostView : public ConstView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = ConstHostView<T, DIM - 1>;

  ConstHostView() : BaseType(){};

  ConstHostView(const HostView<T, DIM>& v) : BaseType(v){};

  explicit ConstHostView(const ConstView<T, DIM>& v) : BaseType(v){};

  ConstHostView(std::shared_ptr<Allocator> alloc, const IndexType& shape)
      : BaseType(std::move(alloc), shape){};

  ConstHostView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  inline auto operator[](const IndexType& index) const -> const T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return this->constPtr_[view_index(index, this->strides_)];
  }

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> ConstHostView<T, DIM> {
    return this->template sub_view_impl<ConstHostView<T, DIM>>(offset, shape);
  }
};


#if defined(BIPP_CUDA) || defined(BIPP_ROCM)

template <typename T, std::size_t DIM>
class DeviceView : public View<T, DIM> {
public:
  using ValueType = T;
  using BaseType = View<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = DeviceView<T, DIM - 1>;

  DeviceView() : BaseType(){};

  explicit DeviceView(const View<T, DIM>& v) : BaseType(v){};

  DeviceView(T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> DeviceView<T, DIM> {
    return this->template sub_view_impl<DeviceView<T, DIM>>(offset, shape);
  }

protected:
  friend ConstView<T, DIM>;
  friend ConstView<T, DIM + 1>;

  DeviceView(const ConstView<T, DIM>& b) : BaseType(b){};
};

template <typename T, std::size_t DIM>
class ConstDeviceView : public ConstView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = ConstDeviceView<T, DIM - 1>;

  ConstDeviceView() : BaseType(){};

  ConstDeviceView(const DeviceView<T, DIM>& v) : BaseType(v){};

  explicit ConstDeviceView(const ConstView<T, DIM>& v) : BaseType(v){};

  ConstDeviceView(std::shared_ptr<Allocator> alloc, const IndexType& shape)
      : BaseType(std::move(alloc), shape){};

  ConstDeviceView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> ConstDeviceView<T, DIM> {
    return this->template sub_view_impl<ConstDeviceView<T, DIM>>(offset, shape);
  }
};

#endif

}  // namespace bipp
