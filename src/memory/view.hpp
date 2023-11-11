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

namespace bipp {

namespace impl {

template <std::size_t DIM>
struct ViewIndexTypeHelper {
  using type = std::array<std::size_t, DIM>;
};

template <>
struct ViewIndexTypeHelper<1> {
  using type = std::size_t;
};

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

template <std::size_t DIM>
using ViewIndexType = typename impl::ViewIndexTypeHelper<DIM>::type;

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
class ViewBase {
public:
  using IndexType = ViewIndexType<DIM>;

  ViewBase() {
    if constexpr(DIM==1) {
      shape_ = 0;
      strides_ = 1;
    } else {
      shape_.fill(0);
      strides_.fill(1);
    }
  }

  ViewBase(const T* ptr, const IndexType& shape, const IndexType& strides)
      : shape_(shape), strides_(strides), totalSize_(view_size(shape)), constPtr_(ptr) {
#ifndef NDEBUG
    assert(this->strides(0) == 1);
    for (std::size_t i = 1; i < DIM; ++i) {
      assert(this->strides(i) >= this->shape(i - 1) * this->strides(i - 1));
    }
#endif
  }

  virtual ~ViewBase() = default;

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

protected:
  friend ViewBase<T, DIM + 1>;

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

    return SLICE_TYPE{ViewBase<T, DIM - 1>{this->constPtr_ + outer_index * this->strides(DIM - 1),
                                           sliceShape, sliceStrides}};
  }

  template <typename VIEW_TYPE>
  auto sub_view_impl(const IndexType& offset, const IndexType& shape) const -> VIEW_TYPE {
    if (!compare_elements(offset, shape_, std::less{}) ||
        !compare_elements(shape, shape_, std::less_equal{}))
      throw InternalError("Sub view offset or shape out of range.");
    return VIEW_TYPE{ViewBase{constPtr_ + view_index(offset, strides_), shape, strides_}};
  }

  IndexType shape_;
  IndexType strides_;
  std::size_t totalSize_ = 0;
  const T* constPtr_ = nullptr;
};

template <typename T, std::size_t DIM>
class ConstHostView : public ViewBase<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ViewBase<T, DIM>;
  using IndexType = ViewIndexType<DIM>;
  using SliceType = ConstHostView<T, DIM - 1>;

  ConstHostView() : BaseType(){};

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

protected:
  friend ViewBase<T, DIM>;
  friend ViewBase<T, DIM + 1>;

  ConstHostView(BaseType&& b) : BaseType(std::move(b)){};
};

template <typename T, std::size_t DIM>
class HostView : public ConstHostView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstHostView<T, DIM>;
  using IndexType = typename BaseType::IndexType;
  using SliceType = HostView<T, DIM - 1>;

  HostView() : BaseType(){};

  HostView(T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  inline auto data() -> T* { return const_cast<T*>(this->constPtr_); }

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
  friend ViewBase<T, DIM>;
  friend ViewBase<T, DIM + 1>;

  HostView(BaseType&& b) : BaseType(std::move(b)){};
};

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)

template <typename T, std::size_t DIM>
class ConstDeviceView : public ViewBase<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ViewBase<T, DIM>;
  using IndexType = ViewIndexType<DIM>;
  using SliceType = ConstDeviceView<T, DIM - 1>;

  ConstDeviceView() : BaseType(){};

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

protected:
  friend ViewBase<T, DIM>;
  friend ViewBase<T, DIM + 1>;

  ConstDeviceView(BaseType&& b) : BaseType(std::move(b)){};
};

template <typename T, std::size_t DIM>
class DeviceView : public ConstDeviceView<T, DIM> {
public:
  using ValueType = T;
  using BaseType = ConstDeviceView<T, DIM>;
  using IndexType = ViewIndexType<DIM>;
  using SliceType = DeviceView<T, DIM - 1>;

  DeviceView() : BaseType(){};

  DeviceView(T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  inline auto data() -> T* { return const_cast<T*>(this->constPtr_); }

  auto slice_view(std::size_t outer_index) const -> SliceType {
    return this->template slice_view_impl<SliceType>(outer_index);
  }

  auto sub_view(const IndexType& offset, const IndexType& shape) const -> DeviceView<T, DIM> {
    return this->template sub_view_impl<DeviceView<T, DIM>>(offset, shape);
  }

protected:
  friend ViewBase<T, DIM>;
  friend ViewBase<T, DIM + 1>;

  DeviceView(BaseType&& b) : BaseType(std::move(b)){};
};

#endif

}  // namespace bipp
