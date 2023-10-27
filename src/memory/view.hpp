#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>

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

template <MemoryType MEM_TYPE, typename T, std::size_t DIM>
class ViewBase {
public:
  using index_type = std::array<std::size_t, DIM>;

  virtual ~ViewBase() = default;

  auto operator=(const ViewBase&) -> ViewBase& = default;

  auto operator=(ViewBase&& b) -> ViewBase& = default;

  inline auto operator()(const index_type& index) const -> const T& {
    return constPtr_[array_index(index, strides_)];
  }

  inline auto get() const -> const T* { return constPtr_; }

  inline auto shape() const noexcept -> const std::array<std::size_t, DIM> { return shape_; }

  inline auto strides() const noexcept -> const std::array<std::size_t, DIM> { return strides_; }

  inline auto data_handler() const noexcept -> const std::shared_ptr<void>& { return data_; }

  inline auto has_ownership() const noexcept -> bool { return bool(data_); }

  inline auto memory_type() const -> const MemoryType { return MEM_TYPE; }

protected:
  ViewBase() {
    shape_.fill(0);
    strides_.fill(0);
  }

  ViewBase(const index_type& shape, const index_type& strides, std::shared_ptr<void> data,
           const T* constPtr)
      : shape_(shape),
        strides_(strides),
        totalSize_(std::accumulate(shape_.begin(), shape_.end(), std::size_t(0))),
        data_(std::move(data)),
        constPtr_(constPtr) {}

  ViewBase(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : shape_(shape), totalSize_(std::accumulate(shape_.begin(), shape_.end(), std::size_t(0))) {
    if (alloc->type() != MEM_TYPE)
      throw InternalError("View: Memory type and allocator type mismatch.");
    if (totalSize_) {
      auto ptr = alloc->allocate(totalSize_ * sizeof(T));
      data_ = std::shared_ptr<void>(ptr, [alloc = std::move(alloc)](void* p) {
        if (p) alloc->deallocate(p);
      });
    }

    strides_[0] = 1;
    for (std::size_t i = 1; i < DIM; ++i) {
      strides_[i] = shape_[i - 1] * strides_[i - 1];
    }
  }

  ViewBase(const T* ptr, const index_type& shape, const index_type& strides)
      : shape_(shape),
        strides_(strides),
        totalSize_(std::accumulate(shape_.begin(), shape_.end(), std::size_t(0))) {
    if (strides_[0] != 1) throw InternalError("View: First stride entry must be 1.");
    for (std::size_t i = 1; i < DIM; ++i) {
      if (strides_[i] < shape_[i - 1] * strides_[i - 1])
        throw InternalError("View: Invalid strides.");
    }
  }

  ViewBase(const ViewBase&) = default;

  ViewBase(ViewBase&& b) = default;

  template <typename UnaryTransformOp>
  inline auto compare_elements(const index_type& left, const index_type& right,
                               UnaryTransformOp&& op) -> bool {
    return std::transform_reduce(left.begin(), left.end(), right.begin(), true, std::logical_and{},
                                 std::forward<UnaryTransformOp>(op));
  }

  template <typename SLICE_TYPE>
  auto slice_view_impl(std::size_t outer_index, bool copyOwnership) -> SLICE_TYPE {
    if (outer_index >= shape_[DIM - 1]) throw InternalError("View slice index out of range.");

    typename SLICE_TYPE::index_type sliceShape, sliceStrides;
    std::copy(this->shape_.begin(), this->shape_.end() - 1, sliceShape.begin());
    std::copy(this->strides_.begin(), this->strides_.end() - 1, sliceStrides.begin());
    return SLICE_TYPE{ViewBase<MemoryType::Host, T, DIM - 1>{
        sliceShape, sliceStrides, copyOwnership ? this->data_ : std::shared_ptr<void>(),
        this->constPtr_ + outer_index * this->strides_[DIM - 1]}};
  }

  template <typename VIEW_TYPE>
  auto sub_view_impl(const index_type& offset, const index_type& shape, bool copyOwnership)
      -> VIEW_TYPE {
    if (!compare_elements(offset, shape_, std::less{}) ||
        !compare_elements(shape, shape_, std::less_equal{}))
      throw InternalError("Sub view offset or shape out of range.");
    return VIEW_TYPE{ViewBase(shape, strides_, copyOwnership ? data_ : std::shared_ptr<void>(),
                              constPtr_ + array_index(offset, strides_))};
  }

  index_type shape_;
  index_type strides_;
  std::size_t totalSize_;
  std::shared_ptr<void> data_;
  const T* constPtr_ = nullptr;
};

template <typename T, std::size_t DIM>
class HostView : public ViewBase<MemoryType::Host, T, DIM> {
public:
  using base_type = ViewBase<MemoryType::Host, T, DIM>;
  using index_type = typename base_type::index_type;
  using slice_type = HostView<T, DIM - 1>;

  HostView() : base_type(){};

  HostView(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : base_type(std::move(alloc), shape){};

  HostView(T* ptr, const index_type& shape, const index_type& strides)
      : base_type(ptr, shape, strides){};

  inline auto get() const -> T* { return this->constPtr_; }

  inline auto operator[](const index_type& index) const -> const T& {
    return this->constPtr_[array_index(index, this->strides_)];
  }

  inline auto operator[](const index_type& index) -> T& {
    return const_cast<T*>(this->constPtr_)[array_index(index, this->strides_)];
  }

  auto slice_view(std::size_t outer_index, bool copyOwnership) -> slice_type {
    return this->template slice_view_impl<slice_type>(outer_index, copyOwnership);
  }

  auto sub_view(const index_type& offset, const index_type& shape, bool copyOwnership)
      -> HostView<T, DIM> {
    return this->template sub_view_impl<HostView<T, DIM>>(offset, shape, copyOwnership);
  }

private:
  friend base_type;

  HostView(base_type&& b) : base_type(std::move(b)){};
};

template <typename T, std::size_t DIM>
class ConstHostView : public ViewBase<MemoryType::Host, T, DIM> {
public:
  using base_type = ViewBase<MemoryType::Host, T, DIM>;
  using index_type = typename base_type::index_type;
  using slice_type = ConstHostView<T, DIM - 1>;

  ConstHostView() : base_type(){};

  ConstHostView(const HostView<T, DIM>& v) : base_type(static_cast<const base_type&>(v)){};

  ConstHostView(HostView<T, DIM>&& v) : base_type(static_cast<base_type&&>(std::move(v))){};

  ConstHostView(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : base_type(std::move(alloc), shape){};

  ConstHostView(const T* ptr, const index_type& shape, const index_type& strides)
      : base_type(ptr, shape, strides){};

  inline auto operator[](const index_type& index) const -> const T& {
    return this->constPtr_[array_index(index, this->strides_)];
  }

  auto slice_view(std::size_t outer_index, bool copyOwnership) -> slice_type {
    return this->template slice_view_impl<slice_type>(outer_index, copyOwnership);
  }

  auto sub_view(const index_type& offset, const index_type& shape, bool copyOwnership)
      -> ConstHostView<T, DIM> {
    return this->template sub_view_impl<ConstHostView<T, DIM>>(offset, shape, copyOwnership);
  }

private:
  friend base_type;

  ConstHostView(base_type&& b) : base_type(std::move(b)){};
};

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
template <typename T, std::size_t DIM>
class DeviceView : public ViewBase<MemoryType::Device, T, DIM> {
public:
  using base_type = ViewBase<DeviceView<T, DIM>, MemoryType::Device, T, DIM>;
  using index_type = typename base_type::index_type;
  using slice_type = DeviceView<T, DIM - 1>;

  DeviceView() : base_type(){};

  DeviceView(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : base_type(std::move(alloc), shape){};

  DeviceView(T* ptr, const index_type& shape, const index_type& strides)
      : base_type(ptr, shape, strides){};

  inline auto get() const -> T* { return this->constPtr_; }

  auto slice_view(std::size_t outer_index, bool copyOwnership) -> slice_type {
    return this->template slice_view_impl<slice_type>(outer_index, copyOwnership);
  }

  auto sub_view(const index_type& offset, const index_type& shape, bool copyOwnership)
      -> DeviceView<T, DIM> {
    return this->template sub_view_impl<DeviceView<T, DIM>>(offset, shape, copyOwnership);
  }

private:
  friend base_type;

  DeviceView(base_type&& b) : base_type(std::move(b)){};
};

template <typename T, std::size_t DIM>
class ConstDeviceView : public ViewBase<MemoryType::Device, T, DIM> {
public:
  using base_type = ViewBase<ConstDeviceView<T, DIM>, MemoryType::Device, T, DIM>;
  using index_type = typename base_type::index_type;
  using slice_type = ConstDeviceView<T, DIM - 1>;

  ConstDeviceView() : base_type(){};

  ConstDeviceView(const DeviceView<T, DIM>& v) : base_type(static_cast<const base_type&>(v)){};

  ConstDeviceView(DeviceView<T, DIM>&& v) : base_type(static_cast<base_type&&>(std::move(v))){};

  ConstDeviceView(std::shared_ptr<Allocator> alloc, const index_type& shape)
      : base_type(std::move(alloc), shape){};

  ConstDeviceView(const T* ptr, const index_type& shape, const index_type& strides)
      : base_type(ptr, shape, strides){};

  auto slice_view(std::size_t outer_index, bool copyOwnership) -> slice_type {
    return this->template slice_view_impl<slice_type>(outer_index, copyOwnership);
  }

  auto sub_view(const index_type& offset, const index_type& shape, bool copyOwnership)
      -> ConstDeviceView<T, DIM> {
    return this->template sub_view_impl<ConstDeviceView<T, DIM>>(offset, shape, copyOwnership);
  }

private:
  friend base_type;

  ConstDeviceView(base_type&& b) : base_type(std::move(b)){};
};
#endif

void test() {
  auto v = HostView<double, 2>();
  ConstHostView cv(v);
}

}  // namespace bipp
