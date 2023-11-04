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
template <std::size_t DIM, std::size_t N>
struct array_index_helper {
  inline static constexpr auto eval(const std::array<std::size_t, DIM>& indices,
                                    const std::array<std::size_t, DIM>& strides) -> std::size_t {
    return indices[N] * strides[N] + array_index_helper<DIM, N - 1>::eval(indices, strides);
  }
};

template <std::size_t DIM>
struct array_index_helper<DIM, 0> {
  inline static constexpr auto eval(const std::array<std::size_t, DIM>& indices,
                                    const std::array<std::size_t, DIM>&) -> std::size_t {
    static_assert(DIM >= 1);
    return indices[0];
  }
};

template <std::size_t DIM>
struct array_index_helper<DIM, 1> {
  inline static constexpr auto eval(const std::array<std::size_t, DIM>& indices,
                                    const std::array<std::size_t, DIM>& strides) -> std::size_t {
    static_assert(DIM >= 2);
    return indices[0] + indices[1] * strides[1];
  }
};

template <std::size_t DIM>
struct array_index_helper<DIM, 2> {
  inline static constexpr auto eval(const std::array<std::size_t, DIM>& indices,
                                    const std::array<std::size_t, DIM>& strides) -> std::size_t {
    static_assert(DIM >= 3);
    return indices[0] + indices[1] * strides[1] + indices[2] * strides[2];
  }
};

}  // namespace impl

template <std::size_t DIM>
inline constexpr auto array_index(const std::array<std::size_t, DIM>& indices,
                                  const std::array<std::size_t, DIM>& strides) -> std::size_t {
  return impl::array_index_helper<DIM, DIM - 1>::eval(indices, strides);
}

template <typename T, std::size_t DIM>
class ViewBase {
public:
  static_assert(std::is_trivially_destructible_v<T>);
  static_assert(std::is_trivially_copyable_v<T>);

  using IndexType = std::array<std::size_t, DIM>;

  ViewBase() {
    shape_.fill(0);
    strides_.fill(0);
  }

  ViewBase(const T* ptr, const IndexType& shape, const IndexType& strides)
      : shape_(shape),
        strides_(strides),
        totalSize_(std::reduce(shape_.begin(), shape_.end(), std::size_t(1), std::multiplies{})),
        constPtr_(ptr) {
    if (strides_[0] != 1) throw InternalError("View: First stride entry must be 1.");
    for (std::size_t i = 1; i < DIM; ++i) {
      if (strides_[i] < shape_[i - 1] * strides_[i - 1])
        throw InternalError("View: Invalid strides.");
    }
  }


  virtual ~ViewBase() = default;

  inline auto data() const -> const T* { return constPtr_; }

  inline auto size() const noexcept -> std::size_t { return totalSize_; }

  inline auto shape() const noexcept -> const std::array<std::size_t, DIM> { return shape_; }

  inline auto strides() const noexcept -> const std::array<std::size_t, DIM> { return strides_; }


protected:
  friend ViewBase<T, DIM + 1>;

  template <typename UnaryTransformOp>
  inline auto compare_elements(const IndexType& left, const IndexType& right,
                               UnaryTransformOp&& op) const -> bool {
    return std::transform_reduce(left.begin(), left.end(), right.begin(), true, std::logical_and{},
                                 std::forward<UnaryTransformOp>(op));
  }

  template <typename SLICE_TYPE>
  auto slice_view_impl(std::size_t outer_index) const -> SLICE_TYPE {
    if (outer_index >= shape_[DIM - 1]) throw InternalError("View slice index out of range.");

    typename SLICE_TYPE::IndexType sliceShape, sliceStrides;
    std::copy(this->shape_.begin(), this->shape_.end() - 1, sliceShape.begin());
    std::copy(this->strides_.begin(), this->strides_.end() - 1, sliceStrides.begin());
    return SLICE_TYPE{ViewBase<T, DIM - 1>{
        this->constPtr_ + outer_index * this->strides_[DIM - 1], sliceShape, sliceStrides}};
  }

  template <typename VIEW_TYPE>
  auto sub_view_impl(const IndexType& offset, const IndexType& shape) const -> VIEW_TYPE {
    if (!compare_elements(offset, shape_, std::less{}) ||
        !compare_elements(shape, shape_, std::less_equal{}))
      throw InternalError("Sub view offset or shape out of range.");
    return VIEW_TYPE{ViewBase{constPtr_ + array_index(offset, strides_), shape, strides_}};
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
  using IndexType = typename BaseType::IndexType;
  using SliceType = ConstHostView<T, DIM - 1>;

  ConstHostView() : BaseType(){};

  ConstHostView(std::shared_ptr<Allocator> alloc, const IndexType& shape)
      : BaseType(std::move(alloc), shape){};

  ConstHostView(const T* ptr, const IndexType& shape, const IndexType& strides)
      : BaseType(ptr, shape, strides){};

  inline auto operator[](const IndexType& index) const -> const T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return this->constPtr_[array_index(index, this->strides_)];
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
    return this->constPtr_[array_index(index, this->strides_)];
  }

  inline auto operator[](const IndexType& index) -> T& {
    assert(this->template compare_elements(index, this->shape_, std::less{}));
    return const_cast<T*>(this->constPtr_)[array_index(index, this->strides_)];
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
        std::memset(this->data(), 0, this->shape_[0] * sizeof(T));
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
  using IndexType = typename BaseType::IndexType;
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
  using IndexType = typename BaseType::IndexType;
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
