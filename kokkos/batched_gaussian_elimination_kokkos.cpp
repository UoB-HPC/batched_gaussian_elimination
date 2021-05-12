
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <Kokkos_Core.hpp>

void kernel(const int buckets, const int nnodes, Kokkos::View<double**,Kokkos::LayoutLeft>& x);
void kernel_shared(const int buckets, const int nnodes, Kokkos::View<double***,Kokkos::LayoutLeft>& A, Kokkos::View<double**,Kokkos::LayoutLeft>& x, Kokkos::View<double**,Kokkos::LayoutLeft>& b);


int main(int argc, char *argv[]) {

  Kokkos::initialize(argc, argv);

  // Print banner
  std::cout
    << std::endl
    << "----------------------------------------------------------" << std::endl
    << "Batched matrix gaussian elimination memory allocation test" << std::endl
    << std::endl
    << "  Kokkos" << std::endl
    << "----------------------------------------------------------" << std::endl
    << std::endl;

  // Check for input arguments
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: " << argv[0] << " order buckets ntimes [option]" << std::endl;
    std::cerr << "  option = 1 2 or 3 runs that case only. Default run all." << std::endl;
    exit(EXIT_FAILURE);
  }

  // Save element order
  int order = atoi(argv[1]);
  if (order < 1) {
    std::cerr << "Error: element order must be 1 or greater" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "Element order: " << order << std::endl;

  // Calculate number of nodes from order
  int nnodes = (1+order)*(1+order)*(1+order);
  std::cout << "Number of nodes/matrix size: " << nnodes << std::endl;

  // Save number of parallel buckets
  int buckets = atoi(argv[2]);
  std::cout << "Number of buckets: " << buckets << std::endl;

  // Save number of times to run solve
  int ntimes = atoi(argv[3]);
  std::cout << "Number of iterations: " << ntimes << std::endl;


  bool run[4] = {false};
  // Check if only running one case
  if (argc == 5) {
    int option = atoi(argv[4]);
    if (option < 1 || option > 3 || option == 2) {
      std::cerr << "Input error: option must be 1 or 3" << std::endl;
      exit(EXIT_FAILURE);
    }
    run[option] = true;
  } else {
    for (int i = 0; i < 4; ++i) {
      run[i] = true;
    }
    run[2] = false;
  }

  // Standard timers
  double time;
  double alloc_time[4];
  double init_time[4];
  double run_time[4];

  //
  // Option 1:
  //   Each thread allocates its own local data
  //
  if (run[1]) {
  int opt = 1;

  // Memory allocation just need x in global
  auto tic = std::chrono::high_resolution_clock::now();
  Kokkos::View<double**, Kokkos::LayoutLeft>  x {"x", static_cast<size_t>(nnodes), static_cast<size_t>(buckets)};
  Kokkos::fence();
  auto toc = std::chrono::high_resolution_clock::now();
  alloc_time[opt] = static_cast<std::chrono::duration<double>>(toc-tic).count();


  // Initialise data occurs later
  init_time[opt] = 0.0;

  // Solve system in each thread a number of times
  tic = std::chrono::high_resolution_clock::now();
  for (int n = 0; n < ntimes; ++n) {
    kernel(buckets, nnodes, x);
  }
  Kokkos::fence();
  toc = std::chrono::high_resolution_clock::now();
  run_time[opt] = static_cast<std::chrono::duration<double>>(toc-tic).count();

 
  }

  //
  // Option 2:
  //   Like Option 1, each thread has its own local data, but initial allocation is done by master
  //
  if (run[2]) {
    std::cout << "TODO" << std::endl;
  /*
  int opt = 2;
  double ** __restrict A = new double*[nthreads];
  double ** __restrict b = new double*[nthreads];
  double ** __restrict x = new double*[nthreads];

  // Allocate array per thread
  time = omp_get_wtime();
  for (int t = 0; t < nthreads; ++t) {
    A[t] = new double[nnodes*nnodes];
    b[t] = new double[nnodes];
    x[t] = new double[nnodes];
  }
  alloc_time[opt] = omp_get_wtime() - time;

  // Initialise data
  time = omp_get_wtime();
  #pragma omp parallel
  {
    for (int j = 0; j < nnodes; ++j) {
      for (int i = 0; i < nnodes; ++i) {
        A[omp_get_thread_num()][i+nnodes*j] = (double)(i+nnodes*j);
      }
      b[omp_get_thread_num()][j] = (double)j;
      x[omp_get_thread_num()][j] = 0.0;
    }
  }
  init_time[opt] = omp_get_wtime() - time;

  // Solve system in each thread a number of times
  time = omp_get_wtime();
  for (int n = 0; n < ntimes; ++n) {
    #pragma omp parallel for
    for (int nb = 0; nb < buckets; ++nb) {
      kernel(nnodes, A[omp_get_thread_num()], x[omp_get_thread_num()], b[omp_get_thread_num()]);
    }
  }
  run_time[opt] = omp_get_wtime() - time;

 
  // Free memory
  #pragma omp parallel
  {
    delete[] A[omp_get_thread_num()];
    delete[] b[omp_get_thread_num()];
    delete[] x[omp_get_thread_num()];
  }
  delete[] A;
  delete[] b;
  delete[] x;
  */
  }



  //
  // Option 3:
  //   Data for all threads allocated in shared array
  //
  if (run[3]) {
  int opt = 3;
  // Allocate arrays
  auto tic = std::chrono::high_resolution_clock::now();
  Kokkos::View<double***, Kokkos::LayoutLeft> A {"A", static_cast<size_t>(nnodes), static_cast<size_t>(nnodes), static_cast<size_t>(buckets)};
  Kokkos::View<double**, Kokkos::LayoutLeft>  b {"b", static_cast<size_t>(nnodes), static_cast<size_t>(buckets)};
  Kokkos::View<double**, Kokkos::LayoutLeft>  x {"x", static_cast<size_t>(nnodes), static_cast<size_t>(buckets)};
  auto toc = std::chrono::high_resolution_clock::now();
  alloc_time[opt] = static_cast<std::chrono::duration<double>>(toc-tic).count();

  // Initialise data
  tic = std::chrono::high_resolution_clock::now();
  // Launch a team for each bucket, each of size nnodes
  Kokkos::TeamPolicy<> policy(buckets, Kokkos::AUTO);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const auto& team_member) {
    const int t = team_member.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, nnodes), [=](const int& j) {
      for (int i = 0; i < nnodes; ++i) {
        A(i,j,t) = static_cast<double>(i+nnodes*j);
      }
      b(j,t) = static_cast<double>(j);
      x(j,t) = 0.0;
    });
  });
  Kokkos::fence();
  toc = std::chrono::high_resolution_clock::now();
  init_time[opt] = static_cast<std::chrono::duration<double>>(toc-tic).count();

  // Solve system in each thread a number of times
  tic = std::chrono::high_resolution_clock::now();
  for (int n = 0; n < ntimes; ++n) {
    // Moved parallelism over bucket inside the kernel function
    kernel_shared(buckets, nnodes, A, x, b);
  }
  Kokkos::fence();
  toc = std::chrono::high_resolution_clock::now();
  run_time[opt] = static_cast<std::chrono::duration<double>>(toc-tic).count();
 
  }

  std::cout
    << "----------" << std::endl;

  for (int opt = 1; opt < 4; ++opt) {
    if (run[opt]) {
      std::cout << std::scientific
        << "Option " << opt << ":" << std::endl
      << "  Allocation: " << alloc_time[opt] << std::endl
      << "  Init:       " << init_time[opt] << std::endl
      << "  Compute:    " << run_time[opt] << std::endl
      << std::endl;
    }
  }
  std::cout << std::endl;

  Kokkos::finalize();

  return EXIT_SUCCESS;

}

// Perform in-place Gaussian elimination on one batch of matrices A to solve Ax=b.
// The matrix is in row-major order
void kernel(const int buckets, const int nnodes, Kokkos::View<double**,Kokkos::LayoutLeft>& x) {
  Kokkos::TeamPolicy<> policy(buckets, Kokkos::AUTO);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(sizeof(double)*nnodes*nnodes+sizeof(double)*nnodes));
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const auto& team_member) {
    const int tid = team_member.league_rank();

    // Allocate team-local A and b Views
    //double* A = (double*) team_member.team_shmem().get_shmem(sizeof(double)*nnodes*nnodes);
    //double* b = (double*) team_member.team_shmem().get_shmem(sizeof(double)*nnodes);
    Kokkos::View<double**,Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> A(team_member.team_scratch(0), static_cast<size_t>(nnodes),  static_cast<size_t>(nnodes));
    Kokkos::View<double*,Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> b(team_member.team_scratch(0), static_cast<size_t>(nnodes));

    // (Re)init
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, nnodes), [=](const int& j) {
      for (int i = 0; i < nnodes; ++i) {
        A(i,j) = (double)(i+nnodes*j);
      }
      b(j) = (double)j;
      x(j, tid) = 0.0;
    });
    team_member.team_barrier();

    //
    // Uses Gaussian elimination
    //
    // Generate upper triangular matrix
    // Subtract multiples of rows from top to bottom
    for (int j = 0; j < nnodes; ++j) { // Loop over rows
      const double Ajj = A(j,j);
      if (Ajj != 0.0) {
        for (int i = j+1; i < nnodes; ++i) { // Loop over rows beneath jth row
          const double c =  A(j,i) / Ajj;
          team_member.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nnodes), [=](const int& k) {
            // Loop over entries in row
            A(k,i) -= c * A(k,j);
          });
          b(i) -= c * b(j);
          team_member.team_barrier();
        }
      }
    }

    // Backwards substitution
    for (int j = nnodes-1; j >= 0; --j) {
      x(j,tid) = b(j) / A(j,j);
      team_member.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nnodes), [=](const int& i) {
        b(i) -= x(j,tid) * A(j,i);
        A(j,i) = 0.0;
      });
      team_member.team_barrier();
    }
  });
}


// Perform in-place Gaussian elimination on ~one~ a bucket of matrices A to solve Ax=b.
// The matrix is in row-major order
// The storage is shared between all threads so has an extra index for thread number
void kernel_shared(const int buckets, const int nnodes, Kokkos::View<double***,Kokkos::LayoutLeft>& A, Kokkos::View<double**,Kokkos::LayoutLeft>& x, Kokkos::View<double**,Kokkos::LayoutLeft>& b) {
  Kokkos::TeamPolicy<> policy(buckets, Kokkos::AUTO);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const auto& team_member) {
    const int tid = team_member.league_rank();

    // Reinit
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member, nnodes), [=](const int& j) {
      for (int i = 0; i < nnodes; ++i) {
        A(i,j,tid) = (double)(i+nnodes*j);
      }
      b(j, tid) = (double)j;
      x(j, tid) = 0.0;
    });
    team_member.team_barrier();

    //
    // Uses Gaussian elimination
    //
    // Generate upper triangular matrix
    // Subtract multiples of rows from top to bottom
    for (int j = 0; j < nnodes; ++j) { // Loop over rows
      const double Ajj = A(j,j,tid);
      if (Ajj != 0.0) {
        for (int i = j+1; i < nnodes; ++i) { // Loop over rows beneath jth row
          const double c =  A(j,i,tid) / Ajj;
          team_member.team_barrier();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nnodes), [=](const int& k) {
            // Loop over entries in row
            A(k,i,tid) -= c * A(k,j,tid);
          });
          b(i,tid) -= c * b(j,tid);
          team_member.team_barrier();
        }
      }
    }

    // Backwards substitution
    for (int j = nnodes-1; j >= 0; --j) {
      x(j,tid) = b(j,tid) / A(j,j,tid);
      team_member.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, nnodes), [=](const int& i) {
        b(i,tid) -= x(j,tid) * A(j,i,tid);
        A(j,i,tid) = 0.0;
      });
      team_member.team_barrier();
    }
  });
}



