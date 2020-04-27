
#include <cstdlib>
#include <iostream>

#include <omp.h>

void kernel(const int nnodes, double * __restrict A, double * __restrict x, double * __restrict b);
void kernel_shared(const int tid, const int nnodes, double * __restrict A, double * __restrict x, double * __restrict b);


int main(int argc, char *argv[]) {

  // Print banner
  std::cout
    << std::endl
    << "----------------------------------------------------------" << std::endl
    << "Batched matrix gaussian elimination memory allocation test" << std::endl
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

  // Discover number of OpenMP threads used
  int nthreads = 1;
  #pragma omp parallel
  {
    #pragma omp master
    nthreads = omp_get_num_threads();
  }
  std::cout << "Number of threads: " << nthreads << std::endl;

  bool run[4] = {false};
  // Check if only running one case
  if (argc == 5) {
    int option = atoi(argv[4]);
    if (option < 1 || option > 3) {
      std::cerr << "Input error: option must be 1, 2 or 3" << std::endl;
    }
    run[option] = true;
  } else {
    for (int i = 0; i < 4; ++i) {
      run[i] = true;
    }
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
  double ** A = new double*[nthreads];
  double ** b = new double*[nthreads];
  double ** x = new double*[nthreads];

  // Allocate array per thread
  time = omp_get_wtime();
  #pragma omp parallel
  {
    A[omp_get_thread_num()] = new double[nnodes*nnodes];
    b[omp_get_thread_num()] = new double[nnodes];
    x[omp_get_thread_num()] = new double[nnodes];
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
  }

  //
  // Option 2:
  //   Like Option 1, each thread has its own local data, but initial allocation is done by master
  //
  if (run[2]) {
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
  }



  //
  // Option 3:
  //   Data for all threads allocated in shared array
  //
  if (run[3]) {
  int opt = 3;
  // Allocate arrays
  time = omp_get_wtime();
  double * __restrict A = new double[nnodes*nnodes*nthreads];
  double * __restrict b = new double[nnodes*nthreads];
  double * __restrict x = new double[nnodes*nthreads];
  alloc_time[opt] = omp_get_wtime() - time;

  // Initialise data
  time = omp_get_wtime();
  #pragma omp parallel
  {
    int t = omp_get_thread_num();
    for (int j = 0; j < nnodes; ++j) {
      for (int i = 0; i < nnodes; ++i) {
        A[i+nnodes*j+nnodes*nnodes*t] = (double)(i+nnodes*j);
      }
      b[j+nnodes*t] = (double)j;
      x[j+nnodes*t] = 0.0;
    }
  }
  init_time[opt] = omp_get_wtime() - time;

  // Solve system in each thread a number of times
  time = omp_get_wtime();
  for (int n = 0; n < ntimes; ++n) {
    #pragma omp parallel for
    for (int nb = 0; nb < buckets; ++nb) {
      kernel_shared(omp_get_thread_num(), nnodes, A, x, b);
    }
  }
  run_time[opt] = omp_get_wtime() - time;

 
  // Free memory
  delete[] A;
  delete[] b;
  delete[] x;
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

  return EXIT_SUCCESS;

}

// Perform in-place Gaussian elimination on one matrix A to solve Ax=b.
// The matrix is in row-major order
void kernel(const int nnodes, double * __restrict A, double * __restrict x, double * __restrict b) {

// Reinit
    for (int j = 0; j < nnodes; ++j) {
      for (int i = 0; i < nnodes; ++i) {
        A[i+nnodes*j] = (double)(i+nnodes*j);
      }
      b[j] = (double)j;
      x[j] = 0.0;
    }

  //
  // Uses Gaussian elimination
  //
  // Generate upper triangular matrix
  // Subtract multiples of rows from top to bottom
  for (int j = 0; j < nnodes; ++j) { // Loop over rows
    const double Ajj = A[j+nnodes*j];
    if (Ajj != 0.0) {
      for (int i = j+1; i < nnodes; ++i) { // Loop over rows beneath jth row
        const double c = A[j+nnodes*i] / Ajj;
        #pragma omp simd
        for (int k = 0; k < nnodes; ++k) { // Loop over entries in row
            A[k+nnodes*i] -= c * A[k+nnodes*j];
        }
        b[i] -= c * b[j];
      }
    }
  }

  // Backwards substitution
  for (int j = nnodes-1; j >= 0; --j) {
    x[j] = b[j] / A[j+nnodes*j];
    #pragma omp simd
    for (int i = 0; i < j; ++i) {
      b[i] -= x[j] * A[j+nnodes*i];
      A[j+nnodes*i] = 0.0;
    }
  }
}

// Perform in-place Gaussian elimination on one matrix A to solve Ax=b.
// The matrix is in row-major order
// The matrix is shared between all threads so has an extra index for thread number
void kernel_shared(const int tid, const int nnodes, double * __restrict A, double * __restrict x, double * __restrict b) {

  // Reinit
    for (int j = 0; j < nnodes; ++j) {
      for (int i = 0; i < nnodes; ++i) {
        A[i+nnodes*j+nnodes*nnodes*tid] = (double)(i+nnodes*j);
      }
      b[j+nnodes*tid] = (double)j;
      x[j+nnodes*tid] = 0.0;
    }
  //
  // Uses Gaussian elimination
  //
  // Generate upper triangular matrix
  // Subtract multiples of rows from top to bottom
  for (int j = 0; j < nnodes; ++j) { // Loop over rows
    const double Ajj = A[j+nnodes*j+nnodes*nnodes*tid];
    if (Ajj != 0.0) {
      for (int i = j+1; i < nnodes; ++i) { // Loop over rows beneath jth row
        const double c =  A[j+nnodes*i+nnodes*nnodes*tid] / Ajj;
        #pragma omp simd
        for (int k = 0; k < nnodes; ++k) { // Loop over entries in row
            A[k+nnodes*i+nnodes*nnodes*tid] -= c * A[k+nnodes*j+nnodes*nnodes*tid];
        }
        b[i+nnodes*tid] -= c * b[j+nnodes*tid];
      }
    }
  }

  // Backwards substitution
  for (int j = nnodes-1; j >= 0; --j) {
    x[j+nnodes*tid] = b[j+nnodes*tid] / A[j+nnodes*j+nnodes*nnodes*tid];
    #pragma omp simd
    for (int i = 0; i < j; ++i) {
      b[i+nnodes*tid] -= x[j+nnodes*tid] * A[j+nnodes*i+nnodes*nnodes*tid];
      A[j+nnodes*i+nnodes*nnodes*tid] = 0.0;
    }
  }
}

