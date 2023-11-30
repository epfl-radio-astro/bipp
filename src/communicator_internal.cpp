#include "communicator_internal.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <optional>
#include <vector>

#include "bipp/bipp.h"
#include "bipp/config.h"

#ifdef BIPP_MPI
#include "context_internal.hpp"
#include "host/nufft_synthesis.hpp"
#include "host/standard_synthesis.hpp"
#include "memory/array.hpp"
#include "memory/copy.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_comm_handle.hpp"
#include "mpi_util/mpi_data_type.hpp"
#include "mpi_util/mpi_init_guard.hpp"
#include "synthesis_factory.hpp"
#include "synthesis_interface.hpp"

#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
#include "gpu/nufft_synthesis.hpp"
#include "gpu/standard_synthesis.hpp"
#include "gpu/util/device_accessor.hpp"
#endif

namespace bipp {

namespace {
struct SerializedPartition {
  std::size_t index =0 ;
  std::size_t x = 0;
  std::size_t y = 0;
  std::size_t z = 0;

  SerializedPartition() = default;
  SerializedPartition(const Partition& p);

  auto get() const -> Partition;
};

struct SerializedNufftOptions {
  decltype(NufftSynthesisOptions::tolerance) tolerance = 0;;
  decltype(NufftSynthesisOptions::collectMemory) collectMemory = 0;
  SerializedPartition localImagePartition;
  SerializedPartition localUVWPartition;

  SerializedNufftOptions() = default;
  SerializedNufftOptions(const NufftSynthesisOptions& opt);

  auto get() const -> NufftSynthesisOptions;
};

struct StatusMessage {
  std::size_t messageIndex = 0;

  struct RootCommDestroyed {
    constexpr static std::size_t index = 1;
  } destroyed;
  struct CreateSynthesis {
    constexpr static std::size_t index = 2;
    SynthesisType synthType;
    std::size_t id, nLevel, nFilter, typeSize;
    SerializedNufftOptions opt;
  } create;
  struct SynthesisCollect {
    constexpr static std::size_t index = 3;
    std::size_t id, nAntenna, nEig;
    double wl;
  } step;
  struct GatherImage {
    constexpr static std::size_t index = 4;
    std::size_t id, idxFilter;
  } gather;

  static auto send_create_standard_synthesis(const MPICommHandle& comm, std::size_t id,
                                             std::size_t nLevel, std::size_t nFilter,
                                             std::size_t typeSize) -> void;

  static auto send_create_nufft_synthesis(const MPICommHandle& comm, std::size_t id,
                                          std::size_t nLevel, std::size_t nFilter,
                                          std::size_t typeSize, const NufftSynthesisOptions& opt)
      -> void;

  static auto send_synthesis_collect(const MPICommHandle& comm, std::size_t id,
                                     std::size_t nAntenna, std::size_t nEig) -> void;

  static auto send_gather_image(const MPICommHandle& comm, std::size_t id, std::size_t idxFilter)
      -> void;

  static auto send_root_comm_destroyed(const MPICommHandle& comm) -> void;

  static auto recv(const MPICommHandle& comm)
      -> std::variant<RootCommDestroyed, CreateSynthesis, SynthesisCollect, GatherImage>;
};

auto send_message(const MPICommHandle& comm, StatusMessage& m) {
  assert(comm.rank() == 0);
  mpi_check_status(MPI_Bcast(&m, sizeof(decltype(m)), MPI_BYTE, 0, comm.get()));
}

auto StatusMessage::send_create_standard_synthesis(const MPICommHandle& comm, std::size_t id,
                                                   std::size_t nLevel, std::size_t nFilter,
                                                   std::size_t typeSize) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::CreateSynthesis::index;

  m.create.synthType = SynthesisType::Standard;
  m.create.id = id;
  m.create.nLevel = nLevel;
  m.create.nFilter = nFilter;
  m.create.typeSize = typeSize;

  send_message(comm, m);
}

auto StatusMessage::send_create_nufft_synthesis(const MPICommHandle& comm, std::size_t id,
                                                 std::size_t nLevel,
                                                std::size_t nFilter, std::size_t typeSize,
                                                const NufftSynthesisOptions& opt) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::CreateSynthesis::index;

  m.create.synthType = SynthesisType::NUFFT;
  m.create.id = id;
  m.create.nLevel = nLevel;
  m.create.nFilter = nFilter;
  m.create.typeSize = typeSize;
  m.create.opt = opt;

  send_message(comm, m);
}

auto StatusMessage::send_synthesis_collect(const MPICommHandle& comm, std::size_t id,
                                        std::size_t nAntenna,  std::size_t nEig)
    -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::SynthesisCollect::index;

  m.step.id = id;
  m.step.nAntenna = nAntenna;
  m.step.nEig = nEig;

  send_message(comm, m);
}

auto StatusMessage::send_root_comm_destroyed(const MPICommHandle& comm) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::RootCommDestroyed::index;

  send_message(comm, m);
}

auto StatusMessage::send_gather_image(const MPICommHandle& comm, std::size_t id,
                                      std::size_t idxFilter) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::GatherImage::index;
  m.gather.id = id;
  m.gather.idxFilter = idxFilter;

  send_message(comm, m);
}

auto StatusMessage::recv(const MPICommHandle& comm)
    -> std::variant<RootCommDestroyed, CreateSynthesis, SynthesisCollect, GatherImage> {
  StatusMessage m;
  mpi_check_status(MPI_Bcast(&m, sizeof(decltype(m)), MPI_BYTE, 0, comm.get()));

  assert(m.messageIndex != 0);

  switch (m.messageIndex) {
    case StatusMessage::RootCommDestroyed::index:
      return m.destroyed;
    case StatusMessage::CreateSynthesis::index:
      return m.create;
    case StatusMessage::SynthesisCollect::index:
      return m.step;
    case StatusMessage::GatherImage::index:
      return m.gather;
    default:
      throw InternalError("Invalid MPI message index");
  }

  return m.destroyed;
}

template <typename T>
auto send_synthesis_init_data(const MPICommHandle& comm, ConstHostView<BippFilter, 1> filter,
                              ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                              ConstView<T, 1> pixelZ, ConstHostView<PartitionGroup, 1> groups)
    -> void {

  assert(comm.rank() == 0);
  // Scatter group to each worker
  {
    std::vector<int> sendCounts(comm.size(), static_cast<int>(sizeof(PartitionGroup)));
    sendCounts[0] = 0;
    std::vector<int> displs(comm.size(), 0);
    for(std::size_t i = 1; i < comm.size(); ++i) {
      displs[i] = displs[i -1] + sendCounts[i - 1];
    }

    PartitionGroup myGroup;
    mpi_check_status(MPI_Scatterv(groups.data(), sendCounts.data(), displs.data(), MPI_BYTE,
                                  &myGroup, 0, MPI_BYTE, 0, comm.get()));
  }

  std::array<MPI_Request, 4> requests;
  std::array<MPI_Status, 4> statuses;

  // Scatter pixels
  {
    std::array<const T*, 3> pixelPtr = {pixelX.data(), pixelY.data(), pixelZ.data()};
    std::vector<int> sendCounts(comm.size(), 0);
    std::vector<int> displs(comm.size(), 0);
    for(std::size_t i = 1; i < comm.size(); ++i) {
      sendCounts[i] = static_cast<int>(groups[i - 1].size);
      displs[i] = static_cast<int>(groups[i - 1].begin);
    }

    T dummy;
    for(std::size_t i = 0; i <3; ++i){
      mpi_check_status(MPI_Iscatterv(pixelPtr[i], sendCounts.data(), displs.data(),
                                    MPIType<T>::get(), &dummy, 0, MPIType<T>::get(), 0,
                                    comm.get(), &requests[i]));
    }
  }

  // Broadcast filter
  std::vector<int> filterInt(filter.size());
  for (std::size_t i = 0; i < filter.size(); ++i) filterInt[i] = filter[i];

  mpi_check_status(
      MPI_Ibcast(filterInt.data(), filterInt.size(), MPIType<int>::get(), 0, comm.get(), &requests[3]));

  // Finalize all
  mpi_check_status(MPI_Waitall(requests.size(), requests.data(), statuses.data()));
}

template <typename T>
auto recv_synthesis_init_data(const MPICommHandle& comm, std::shared_ptr<ContextInternal> ctx,
                              const StatusMessage::CreateSynthesis& info)
    -> std::unique_ptr<SynthesisInterface<T>> {
  assert(comm.rank() != 0);

  auto t = ctx->logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "recv init data");

  PartitionGroup myGroup;
  mpi_check_status(MPI_Scatterv(nullptr, nullptr, nullptr, MPI_BYTE, &myGroup,
                                sizeof(PartitionGroup), MPI_BYTE, 0, comm.get()));

  HostArray<T, 2> pixel;
  HostArray<BippFilter, 1> filter;

  if(ctx->processing_unit() == BIPP_PU_CPU) {
    pixel = HostArray<T, 2>(ctx->host_alloc(), {myGroup.size, 3});
    filter = HostArray<BippFilter, 1>(ctx->host_alloc(), info.nFilter);
  }
  else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    pixel = ctx->gpu_queue().create_pinned_array<T, 2>({myGroup.size, 3});
    filter = ctx->gpu_queue().create_host_array<BippFilter, 1>(info.nFilter);
#else
    throw GPUSupportError();
#endif
  }

  std::array<MPI_Request, 4> requests;
  std::array<MPI_Status, 4> statuses;

  // Scatter pixels
  assert(pixel.shape(0) == myGroup.size);
  assert(pixel.shape(1) == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    mpi_check_status(MPI_Iscatterv(nullptr, nullptr, nullptr, MPIType<T>::get(),
                                   pixel.slice_view(i).data(), myGroup.size, MPIType<T>::get(), 0,
                                   comm.get(), &requests[i]));
  }

  // Broadcast filter
  std::vector<int> filterInt(filter.size());
  for (std::size_t i = 0; i < filter.size(); ++i) filterInt[i] = filter[i];

  mpi_check_status(
      MPI_Ibcast(filterInt.data(), filterInt.size(), MPIType<int>::get(), 0, comm.get(), &requests[3]));

  // Finalize all
  mpi_check_status(MPI_Waitall(requests.size(), requests.data(), statuses.data()));

  for (std::size_t i = 0; i < filter.size(); ++i) filter[i] = BippFilter(filterInt[i]);

  // stop mpi communication timing
  t.stop();

  auto t2 = ctx->logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "distributed synthesis create");

  if (info.synthType == SynthesisType::NUFFT)
    return SynthesisFactory<T>::create_nufft_synthesis(std::move(ctx), info.opt.get(), info.nLevel,
                                                       filter, pixel.slice_view(0),
                                                       pixel.slice_view(1), pixel.slice_view(2));

  return SynthesisFactory<T>::create_standard_synthesis(std::move(ctx), info.nLevel, filter,
                                                        pixel.slice_view(0), pixel.slice_view(1),
                                                        pixel.slice_view(2));

}

template <typename T>
auto broadcast_synthesis_collect_data(const MPICommHandle& comm, HostView<std::complex<T>, 2> vView,
                                   HostView<T, 2> dMasked, HostView<T, 2> xyzUvwView) -> void {
  std::array<MPI_Request, 3> requests;
  std::array<MPI_Status, 3> statuses;

  auto vDataType = MPIDatatypeHandle::create(vView);
  mpi_check_status(MPI_Ibcast(vView.data(), 1, vDataType.get(), 0, comm.get(), &requests[0]));

  auto dMaskedDataType = MPIDatatypeHandle::create(dMasked);
  mpi_check_status(
      MPI_Ibcast(dMasked.data(), 1, dMaskedDataType.get(), 0, comm.get(), &requests[1]));

  auto xyzUvwDataType = MPIDatatypeHandle::create(xyzUvwView);
  mpi_check_status(
      MPI_Ibcast(xyzUvwView.data(), 1, xyzUvwDataType.get(), 0, comm.get(), &requests[2]));

  mpi_check_status(MPI_Waitall(requests.size(), requests.data(), statuses.data()));
}

template <typename T>
auto send_synthesis_collect_data(const MPICommHandle& comm, ConstHostView<std::complex<T>, 2> vView,
                                   ConstHostView<T, 2> dMasked, ConstHostView<T, 2> xyzUvwView) -> void {
  assert(comm.rank() == 0);

  // Broadcast will not modify root data
  broadcast_synthesis_collect_data<T>(
      comm,
      HostView<std::complex<T>, 2>(const_cast<std::complex<T>*>(vView.data()), vView.shape(),
                                   vView.strides()),
      HostView<T, 2>(const_cast<T*>(dMasked.data()), dMasked.shape(), dMasked.strides()),
      HostView<T, 2>(const_cast<T*>(xyzUvwView.data()), xyzUvwView.shape(), xyzUvwView.strides()));
}

template <typename T>
auto recv_synthesis_collect_data(const MPICommHandle& comm,
                                 const StatusMessage::SynthesisCollect& info,
                                 SynthesisInterface<T>& syn) -> void {
  auto& ctx = syn.context();

  auto t =
      ctx.logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, pointer_to_string(&syn) + " recv collect data");

  auto v = HostArray<std::complex<T>, 2>();
  auto dMasked = HostArray<T, 2>();
  auto xyzUvw = HostArray<T, 2>();

  if (ctx.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctx.gpu_queue();
    v = queue.template create_pinned_array<std::complex<T>, 2>({info.nAntenna, info.nEig});
    dMasked = queue.template create_pinned_array<T, 2>({info.nEig, syn.image().shape(1)});
    xyzUvw = queue.template create_pinned_array<T, 2>(
        {syn.type() == SynthesisType::Standard ? info.nAntenna : info.nAntenna * info.nAntenna, 3});
#else
    throw GPUSupportError();
#endif
  } else {
    v = HostArray<std::complex<T>, 2>(ctx.host_alloc(), {info.nAntenna, info.nEig});
    dMasked = HostArray<T, 2>(ctx.host_alloc(), {info.nEig, syn.image().shape(1)});
    xyzUvw = HostArray<T, 2>(
        ctx.host_alloc(),
        {syn.type() == SynthesisType::Standard ? info.nAntenna : info.nAntenna * info.nAntenna, 3});
  }

  broadcast_synthesis_collect_data<T>(comm, v, dMasked, xyzUvw);

  t.stop();

  auto t2 = ctx.logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO,
                                               pointer_to_string(&syn) + " collect");
  syn.collect(info.wl, v, dMasked, xyzUvw);
}


template <typename T>
auto send_img_data(const MPICommHandle& comm,const StatusMessage::GatherImage& info, SynthesisInterface<T>& syn)
    -> void {

  assert(comm.rank() != 0);


  auto& ctx = syn.context();


  HostArray<T, 2> imgArray;

  auto t1 = ctx.logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "synthesis get image");
  if (ctx.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctx.gpu_queue();
    imgArray = queue.template create_pinned_array<T, 2>(syn.image().slice_view(0).shape());
    syn.get(syn.filter(info.idxFilter), imgArray);
    queue.sync();
#else
    throw GPUSupportError();
#endif
  } else {
    imgArray = HostArray<T, 2>(ctx.host_alloc(), syn.image().slice_view(0).shape());
    syn.get(syn.filter(info.idxFilter), imgArray);
  }

  t1.stop();
  auto t2 = ctx.logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "distributed image gather");
  T dummy;
  for(std::size_t idxLevel = 0; idxLevel < imgArray.shape(1); ++idxLevel) {
    auto imgSlice = imgArray.slice_view(idxLevel);
    mpi_check_status(MPI_Gatherv(imgSlice.data(), imgSlice.size(), MPIType<T>::get(), &dummy,
                                 nullptr, nullptr, MPIType<T>::get(), 0, comm.get()));
  }
}

template <typename T>
auto recv_img_data(const MPICommHandle& comm, ConstHostView<PartitionGroup, 1> groups,
                   HostView<T, 2> img) -> void {
  assert(comm.rank() == 0);

  std::vector<int> recvCounts(comm.size(), 0);
  std::vector<int> displs(comm.size(), 0);
  for (std::size_t i = 1; i < comm.size(); ++i) {
    recvCounts[i] = static_cast<int>(groups[i - 1].size);
    displs[i] = static_cast<int>(groups[i - 1].begin);
  }

  T dummy;
  for (std::size_t idxLevel = 0; idxLevel < img.shape(1); ++idxLevel) {
    auto imgSlice = img.slice_view(idxLevel);
    mpi_check_status(MPI_Gatherv(&dummy, 0, MPIType<T>::get(), imgSlice.data(), recvCounts.data(),
                                 displs.data(), MPIType<T>::get(), 0, comm.get()));
  }
}

}  // namespace

SerializedPartition::SerializedPartition(const Partition& p) {
  index = p.method.index();

  if (std::holds_alternative<Partition::Grid>(p.method)) {
    const auto& g = std::get<Partition::Grid>(p.method);
    x = g.dimensions[0];
    y = g.dimensions[1];
    z = g.dimensions[2];
  }
}

auto SerializedPartition::get() const -> Partition {
  Partition p;
  p.method = Partition::None{};
  if (p.method.index() == index) return p;
  p.method = Partition::Grid{{x, y, z}};
  if (p.method.index() == index) return p;
  p.method = Partition::Auto{};
  assert(p.method.index() == index);
  return p;
}

SerializedNufftOptions::SerializedNufftOptions(const NufftSynthesisOptions& opt) {
  tolerance = opt.tolerance;
  collectMemory = opt.collectMemory;
  localImagePartition = opt.localImagePartition;
  localUVWPartition = opt.localUVWPartition;
}

auto SerializedNufftOptions::get() const -> NufftSynthesisOptions {
  NufftSynthesisOptions opt;
  opt.tolerance = tolerance;
  opt.collectMemory = collectMemory;
  opt.localImagePartition = localImagePartition.get();
  opt.localUVWPartition = localUVWPartition.get();
  return opt;
}



CommunicatorInternal::CommunicatorInternal(MPI_Comm comm) {
  initialize_mpi_init_guard();

  MPI_Comm newCom;
  mpi_check_status(MPI_Comm_dup(comm, &newCom));
  comm_ = MPICommHandle(newCom);
}

CommunicatorInternal::~CommunicatorInternal() {
  try {
    if (is_root()) {
      StatusMessage::send_root_comm_destroyed(comm_);
    } else if (!rootDestroyed) {
      auto m = StatusMessage::recv(comm_);
      assert(std::holds_alternative<StatusMessage::RootCommDestroyed>(m));
    }
  } catch (...) {
  }
}

auto CommunicatorInternal::attach_non_root(std::shared_ptr<ContextInternal> ctx) -> void {
  if (is_root()) throw InvalidParameterError();

  std::unordered_map<std::size_t, std::unique_ptr<SynthesisInterface<float>>> synthesisFloat;
  std::unordered_map<std::size_t, std::unique_ptr<SynthesisInterface<double>>> synthesisDouble;

  auto t1 = ctx->logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "attach");
  while (!rootDestroyed) {
    auto t2 = ctx->logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "recv status");
    const auto m = StatusMessage::recv(comm_);
    t2.stop();
    std::visit(
        [&](auto&& info) {
          using T = std::decay_t<decltype(info)>;
          if constexpr (std::is_same_v<T, StatusMessage::RootCommDestroyed>) {
            // Root comm destroyed
            rootDestroyed = true;
          } else if constexpr (std::is_same_v<T, StatusMessage::CreateSynthesis>) {
            // Create synthesis
            if (info.typeSize == sizeof(float)) {
              if (synthesisFloat.count(info.id))
                throw InternalError("MPI Create Synthesis: Duplicate synthesis id");
              synthesisFloat.emplace(info.id, recv_synthesis_init_data<float>(comm_, ctx, info));
            } else {
              if (synthesisDouble.count(info.id))
                throw InternalError("MPI Create Synthesis: Duplicate synthesis id");
              synthesisDouble.emplace(info.id, recv_synthesis_init_data<double>(comm_, ctx, info));
            }
          } else if constexpr (std::is_same_v<T, StatusMessage::SynthesisCollect>) {
            // Synthesis collect
            auto itFloat = synthesisFloat.find(info.id);
            if (itFloat != synthesisFloat.end()) {
              recv_synthesis_collect_data<float>(comm_, info, *(itFloat->second));
            } else {
              auto itDouble = synthesisDouble.find(info.id);
              if (itDouble != synthesisDouble.end()) {
                recv_synthesis_collect_data<double>(comm_, info, *(itDouble->second));
              } else
                throw InternalError("MPI Collect step: unknown synthesis id.");
            }
          } else if constexpr (std::is_same_v<T, StatusMessage::GatherImage>) {
            // Gather image on root
            auto itFloat = synthesisFloat.find(info.id);
            if (itFloat != synthesisFloat.end()) {
              send_img_data<float>(comm_, info, *(itFloat->second));
            } else {
              auto itDouble = synthesisDouble.find(info.id);
              if (itDouble != synthesisDouble.end()) {
                send_img_data<double>(comm_, info, *(itDouble->second));
              } else
                throw InternalError("MPI Collect step: unknown synthesis id.");
            }
          }
        },
        m);
  }
}

template <typename T, typename>
auto CommunicatorInternal::send_synthesis_init(std::optional<NufftSynthesisOptions> nufftOpt,
                                               std::size_t nLevel,
                                               ConstHostView<BippFilter, 1> filter,
                                               ConstView<T, 1> pixelX, ConstView<T, 1> pixelY,
                                               ConstView<T, 1> pixelZ,
                                               ConstHostView<PartitionGroup, 1> groups)
    -> std::size_t {
  const auto id = this->generate_local_id();
  if(nufftOpt) {
    StatusMessage::send_create_nufft_synthesis(comm_, id, nLevel, filter.size(), sizeof(T),
                                               nufftOpt.value());
  } else {
    StatusMessage::send_create_standard_synthesis(comm_, id, nLevel, filter.size(), sizeof(T));
  }

  send_synthesis_init_data<T>(comm_, filter, pixelX, pixelY, pixelZ, groups);

  return id;
}

template <typename T, typename>
auto CommunicatorInternal::send_synthesis_collect(std::size_t id,
                                                  ConstHostView<std::complex<T>, 2> vView,
                                                  ConstHostView<T, 2> dMasked,
                                                  ConstHostView<T, 2> xyzUvwView) -> void {
  StatusMessage::send_synthesis_collect(comm_, id, vView.shape(0), vView.shape(1));
  send_synthesis_collect_data<T>(comm_, vView, dMasked, xyzUvwView);
}

template <typename T, typename>
auto CommunicatorInternal::gather_image(std::size_t id, std::size_t idxFilter, ConstHostView<PartitionGroup, 1> groups,
                  HostView<T, 2> img) -> void {
  StatusMessage::send_gather_image(comm_, id, idxFilter);
  recv_img_data<T>(comm_, groups, img);
}

template auto CommunicatorInternal::send_synthesis_init<float>(std::optional<NufftSynthesisOptions> nufftOpt,
    std::size_t nLevel, ConstHostView<BippFilter, 1> filter, ConstView<float, 1> pixelX,
    ConstView<float, 1> pixelY, ConstView<float, 1> pixelZ, ConstHostView<PartitionGroup, 1> groups)
    -> std::size_t;

template auto CommunicatorInternal::send_synthesis_init<double>(std::optional<NufftSynthesisOptions> nufftOpt,
    std::size_t nLevel, ConstHostView<BippFilter, 1> filter, ConstView<double, 1> pixelX,
    ConstView<double, 1> pixelY, ConstView<double, 1> pixelZ,
    ConstHostView<PartitionGroup, 1> groups) -> std::size_t;

template auto CommunicatorInternal::send_synthesis_collect<float>(
    std::size_t id, ConstHostView<std::complex<float>, 2> vView, ConstHostView<float, 2> dMasked,
    ConstHostView<float, 2> xyzUvwView) -> void;

template auto CommunicatorInternal::send_synthesis_collect<double>(
    std::size_t id, ConstHostView<std::complex<double>, 2> vView, ConstHostView<double, 2> dMasked,
    ConstHostView<double, 2> xyzUvwView) -> void;

template auto CommunicatorInternal::gather_image<float>(std::size_t id, std::size_t idxFilter,
                                                        ConstHostView<PartitionGroup, 1> groups,
                                                        HostView<float, 2> img) -> void;
template auto CommunicatorInternal::gather_image<double>(std::size_t id, std::size_t idxFilter,
                                                         ConstHostView<PartitionGroup, 1> groups,
                                                         HostView<double, 2> img) -> void;

}  // namespace bipp
#endif