#include "communicator_internal.hpp"

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "bipp/bipp.h"
#include "bipp/config.h"

#ifdef BIPP_MPI
#include "context_internal.hpp"
#include "host/collector.hpp"
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
#include "gpu/collector.hpp"
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
  std::size_t collectGroupSize = 0;
  SerializedPartition localImagePartition;
  SerializedPartition localUVWPartition;
  bool normalizeImage = true;

  SerializedNufftOptions() = default;
  SerializedNufftOptions(const NufftSynthesisOptions& opt);

  auto get() const -> NufftSynthesisOptions;
};

struct SerializedStandardOptions {
  std::size_t collectGroupSize = 0;
  bool normalizeImage = true;

  SerializedStandardOptions() = default;
  SerializedStandardOptions(const StandardSynthesisOptions& opt);

  auto get() const -> StandardSynthesisOptions;
};

struct StatusMessage {
  std::size_t messageIndex = 0;

  struct RootCommDestroyed {
    constexpr static std::size_t index = 1;
  } destroyed;
  struct CreateSynthesis {
    constexpr static std::size_t index = 2;
    SynthesisType synthType;
    std::size_t id, nLevel, typeSize;
    SerializedNufftOptions optNufft;
    SerializedStandardOptions optStandard;
  } create;
  struct SynthesisCollect {
    constexpr static std::size_t index = 3;
    std::size_t id, bufferSize;
  } step;
  struct GatherImage {
    constexpr static std::size_t index = 4;
    std::size_t id;
  } gather;

  static auto send_create_standard_synthesis(const MPICommHandle& comm, std::size_t id,
                                             std::size_t nLevel, std::size_t typeSize,
                                             const StandardSynthesisOptions& opt) -> void;

  static auto send_create_nufft_synthesis(const MPICommHandle& comm, std::size_t id,
                                          std::size_t nLevel, std::size_t typeSize,
                                          const NufftSynthesisOptions& opt) -> void;

  static auto send_synthesis_collect(const MPICommHandle& comm, std::size_t id,
                                     std::size_t bufferSize) -> void;

  static auto send_gather_image(const MPICommHandle& comm, std::size_t id)
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
                                                   std::size_t nLevel, std::size_t typeSize,
                                                   const StandardSynthesisOptions& opt) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::CreateSynthesis::index;

  m.create.synthType = SynthesisType::Standard;
  m.create.id = id;
  m.create.nLevel = nLevel;
  m.create.typeSize = typeSize;
  m.create.optStandard = opt;

  send_message(comm, m);
}

auto StatusMessage::send_create_nufft_synthesis(const MPICommHandle& comm, std::size_t id,
                                                std::size_t nLevel, std::size_t typeSize,
                                                const NufftSynthesisOptions& opt) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::CreateSynthesis::index;

  m.create.synthType = SynthesisType::NUFFT;
  m.create.id = id;
  m.create.nLevel = nLevel;
  m.create.typeSize = typeSize;
  m.create.optNufft = opt;

  send_message(comm, m);
}

auto StatusMessage::send_synthesis_collect(const MPICommHandle& comm, std::size_t id, std::size_t bufferSize)
    -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::SynthesisCollect::index;

  m.step.id = id;
  m.step.bufferSize = bufferSize;

  send_message(comm, m);
}

auto StatusMessage::send_root_comm_destroyed(const MPICommHandle& comm) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::RootCommDestroyed::index;

  send_message(comm, m);
}

auto StatusMessage::send_gather_image(const MPICommHandle& comm, std::size_t id) -> void {
  StatusMessage m;
  m.messageIndex = StatusMessage::GatherImage::index;
  m.gather.id = id;

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
auto send_synthesis_init_data(const MPICommHandle& comm, ConstView<T, 1> pixelX,
                              ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ,
                              ConstHostView<PartitionGroup, 1> groups) -> void {
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

  std::array<MPI_Request, 3> requests;
  std::array<MPI_Status, 3> statuses;

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

  if(ctx->processing_unit() == BIPP_PU_CPU) {
    pixel = HostArray<T, 2>(ctx->host_alloc(), {myGroup.size, 3});
  }
  else {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    pixel = ctx->gpu_queue().create_pinned_array<T, 2>({myGroup.size, 3});
#else
    throw GPUSupportError();
#endif
  }

  std::array<MPI_Request, 3> requests;
  std::array<MPI_Status, 3> statuses;

  // Scatter pixels
  assert(pixel.shape(0) == myGroup.size);
  assert(pixel.shape(1) == 3);
  for (std::size_t i = 0; i < 3; ++i) {
    mpi_check_status(MPI_Iscatterv(nullptr, nullptr, nullptr, MPIType<T>::get(),
                                   pixel.slice_view(i).data(), myGroup.size, MPIType<T>::get(), 0,
                                   comm.get(), &requests[i]));
  }

  // Finalize all
  mpi_check_status(MPI_Waitall(requests.size(), requests.data(), statuses.data()));

  // stop mpi communication timing
  t.stop();

  auto t2 = ctx->logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "distributed synthesis create");

  if (info.synthType == SynthesisType::NUFFT)
    return SynthesisFactory<T>::create_nufft_synthesis(std::move(ctx), info.optNufft.get(),
                                                       info.nLevel, pixel.slice_view(0),
                                                       pixel.slice_view(1), pixel.slice_view(2));

  return SynthesisFactory<T>::create_standard_synthesis(std::move(ctx), info.optStandard.get(),
                                                        info.nLevel, pixel.slice_view(0),
                                                        pixel.slice_view(1), pixel.slice_view(2));
}

template <typename T>
auto send_synthesis_collect_data(const MPICommHandle& comm, ConstHostView<char, 1> ser) -> void {
  assert(comm.rank() == 0);

  // will not modify input
  // TODO: handle large buffer (larger than int)
  mpi_check_status(
      MPI_Bcast(const_cast<char*>(ser.data()), ser.size(), MPIType<char>::get(), 0, comm.get()));
}

template <typename T>
auto recv_synthesis_collect_data(const MPICommHandle& comm,
                                 const StatusMessage::SynthesisCollect& info,
                                 SynthesisInterface<T>& syn) -> void {
  auto& ctx = syn.context();

  auto t = ctx->logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO,
                                              pointer_to_string(&syn) + " recv collect data");

  std::unique_ptr<CollectorInterface<T>> collector;
  HostArray<char, 1> ser;

  if (ctx->processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctx->gpu_queue();
    collector = std::make_unique<gpu::Collector<T>>(ctx);
    ser = queue.template create_pinned_array<char, 1>(info.bufferSize);
#else
    throw GPUSupportError();
#endif
  } else {
    collector = std::make_unique<host::Collector<T>>(ctx);
    ser = HostArray<char, 1>(ctx->host_alloc(), info.bufferSize);
  }

  // TODO: handle large buffer (larger than int)
  mpi_check_status(
      MPI_Bcast(const_cast<char*>(ser.data()), ser.size(), MPIType<char>::get(), 0, comm.get()));


  t.stop();

  auto t2 = ctx->logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO,
                                                pointer_to_string(&syn) + " collect");

  collector->deserialize(ser);

  syn.process(*collector);
}


template <typename T>
auto send_img_data(const MPICommHandle& comm,const StatusMessage::GatherImage& info, SynthesisInterface<T>& syn)
    -> void {

  assert(comm.rank() != 0);

  auto& ctx = *syn.context();

  HostArray<T, 2> imgArray;

  auto t1 = ctx.logger().measure_scoped_timing(BIPP_LOG_LEVEL_INFO, "synthesis get image");
  if (ctx.processing_unit() == BIPP_PU_GPU) {
#if defined(BIPP_CUDA) || defined(BIPP_ROCM)
    auto& queue = ctx.gpu_queue();
    imgArray = queue.template create_pinned_array<T, 2>(syn.image().shape());
    syn.get(imgArray);
    queue.sync();
#else
    throw GPUSupportError();
#endif
  } else {
    imgArray = HostArray<T, 2>(ctx.host_alloc(), syn.image().shape());
    syn.get(imgArray);
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
  collectGroupSize = opt.collectGroupSize.value_or(0);
  localImagePartition = opt.localImagePartition;
  localUVWPartition = opt.localUVWPartition;
  normalizeImage = opt.normalizeImage;
}

auto SerializedNufftOptions::get() const -> NufftSynthesisOptions {
  NufftSynthesisOptions opt;
  opt.tolerance = tolerance;
  if(collectGroupSize)
    opt.collectGroupSize = collectGroupSize;
  else
    opt.collectGroupSize = std::nullopt;
  opt.localImagePartition = localImagePartition.get();
  opt.localUVWPartition = localUVWPartition.get();
  opt.normalizeImage = normalizeImage;
  return opt;
}

SerializedStandardOptions::SerializedStandardOptions(const StandardSynthesisOptions& opt) {
  collectGroupSize = opt.collectGroupSize.value_or(0);
  normalizeImage = opt.normalizeImage;
}

auto SerializedStandardOptions::get() const -> StandardSynthesisOptions {
  StandardSynthesisOptions opt;
  if(collectGroupSize)
    opt.collectGroupSize = collectGroupSize;
  else
    opt.collectGroupSize = std::nullopt;
  opt.normalizeImage = normalizeImage;
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
auto CommunicatorInternal::send_nufft_synthesis_init(const NufftSynthesisOptions& opt,
                                                     std::size_t nLevel, ConstView<T, 1> pixelX,
                                                     ConstView<T, 1> pixelY, ConstView<T, 1> pixelZ,
                                                     ConstHostView<PartitionGroup, 1> groups)
    -> std::size_t {
  const auto id = this->generate_local_id();
  StatusMessage::send_create_nufft_synthesis(comm_, id, nLevel, sizeof(T), opt);

  send_synthesis_init_data<T>(comm_, pixelX, pixelY, pixelZ, groups);

  return id;
}

template <typename T, typename>
auto CommunicatorInternal::send_standard_synthesis_init(const StandardSynthesisOptions& opt,
                                                        std::size_t nLevel, ConstView<T, 1> pixelX,
                                                        ConstView<T, 1> pixelY,
                                                        ConstView<T, 1> pixelZ,
                                                        ConstHostView<PartitionGroup, 1> groups)
    -> std::size_t {
  const auto id = this->generate_local_id();
  StatusMessage::send_create_standard_synthesis(comm_, id, nLevel, sizeof(T), opt);

  send_synthesis_init_data<T>(comm_, pixelX, pixelY, pixelZ, groups);

  return id;
}

template <typename T, typename>
auto CommunicatorInternal::send_synthesis_collect(std::size_t id,
                                                  const CollectorInterface<T>& collector) -> void {
  auto ser = collector.serialize();
  StatusMessage::send_synthesis_collect(comm_, id, ser.size());
  send_synthesis_collect_data<T>(comm_, ser);
}

template <typename T, typename>
auto CommunicatorInternal::gather_image(std::size_t id, ConstHostView<PartitionGroup, 1> groups,
                  HostView<T, 2> img) -> void {
  StatusMessage::send_gather_image(comm_, id);
  recv_img_data<T>(comm_, groups, img);
}

template auto CommunicatorInternal::send_standard_synthesis_init<float>(
    const StandardSynthesisOptions& opt, std::size_t nLevel, ConstView<float, 1> pixelX,
    ConstView<float, 1> pixelY, ConstView<float, 1> pixelZ, ConstHostView<PartitionGroup, 1> groups)
    -> std::size_t;

template auto CommunicatorInternal::send_standard_synthesis_init<double>(
    const StandardSynthesisOptions& opt, std::size_t nLevel, ConstView<double, 1> pixelX,
    ConstView<double, 1> pixelY, ConstView<double, 1> pixelZ,
    ConstHostView<PartitionGroup, 1> groups) -> std::size_t;

template auto CommunicatorInternal::send_nufft_synthesis_init<float>(
    const NufftSynthesisOptions& opt, std::size_t nLevel, ConstView<float, 1> pixelX,
    ConstView<float, 1> pixelY, ConstView<float, 1> pixelZ, ConstHostView<PartitionGroup, 1> groups)
    -> std::size_t;

template auto CommunicatorInternal::send_nufft_synthesis_init<double>(
    const NufftSynthesisOptions& opt, std::size_t nLevel, ConstView<double, 1> pixelX,
    ConstView<double, 1> pixelY, ConstView<double, 1> pixelZ,
    ConstHostView<PartitionGroup, 1> groups) -> std::size_t;

template auto CommunicatorInternal::send_synthesis_collect<float>(
    std::size_t id, const CollectorInterface<float>& collector) -> void;

template auto CommunicatorInternal::send_synthesis_collect<double>(
    std::size_t id, const CollectorInterface<double>& collector) -> void;

template auto CommunicatorInternal::gather_image<float>(std::size_t id, 
                                                        ConstHostView<PartitionGroup, 1> groups,
                                                        HostView<float, 2> img) -> void;
template auto CommunicatorInternal::gather_image<double>(std::size_t id,
                                                         ConstHostView<PartitionGroup, 1> groups,
                                                         HostView<double, 2> img) -> void;

}  // namespace bipp
#endif
