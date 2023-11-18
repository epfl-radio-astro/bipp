#pragma once

#include <cstring>
#include <memory>

#include "bipp/bipp.h"
#include "bipp/config.h"

#ifdef BIPP_MPI

#include <mpi.h>

#include "mpi_util/mpi_check_status.hpp"

namespace bipp {

// Takes ownerships of a Commnunicator
class MPICommHandle {
public:
  MPICommHandle() : comm_(new MPI_Comm(MPI_COMM_SELF)), size_(1), rank_(0) {}

  explicit MPICommHandle(const MPI_Comm& comm) {
    const MPI_Comm worldComm = MPI_COMM_WORLD;
    const MPI_Comm selfComm = MPI_COMM_SELF;
    if (!std::memcmp(&comm, &worldComm, sizeof(MPI_Comm)) ||
        !std::memcmp(&comm, &selfComm, sizeof(MPI_Comm))) {
      // don't free predifned communicators
      comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(comm));
    } else {
      comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(comm), [](MPI_Comm* ptr) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) {
          MPI_Comm_free(ptr);
        }
        delete ptr;
      });
    }

    mpi_check_status(MPI_Comm_size(*comm_, &size_));
    mpi_check_status(MPI_Comm_rank(*comm_, &rank_));
  }

  inline auto get() const -> const MPI_Comm& { return *comm_; }

  inline auto size() const noexcept -> int { return size_; }

  inline auto rank() const noexcept -> int { return rank_; }

  inline auto clone() const -> MPICommHandle {
    MPI_Comm newComm;
    mpi_check_status(MPI_Comm_dup(this->get(), &newComm));
    return MPICommHandle(newComm);
  }

private:
  std::shared_ptr<MPI_Comm> comm_ = nullptr;
  int size_ = 1;
  int rank_ = 0;
};

}  // namespace bipp

#endif
