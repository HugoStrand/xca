#pragma once

#include <mpi/mpi.hpp>
#include <itertools/itertools.hpp>
#include <stdexcept>

namespace triqs_xca::detail {

  // An implementation detail of topology orchestration. Numerical Backbone
  // execution does not consult MPI, and no collective is performed here.
  inline auto local_work_range(long count) {
    if (mpi::has_env && (!mpi::is_initialized() || mpi::is_finalized()))
      throw std::runtime_error("diagram evaluation requires an initialized MPI runtime under an MPI launcher");
    mpi::communicator comm;
    return itertools::chunk_range(0, count, comm.size(), comm.rank());
  }

} // namespace triqs_xca::detail
