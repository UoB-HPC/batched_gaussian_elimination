# Memory allocation options for batched Gaussian Elimination

The code performs a Gaussian Elimination on a batch of linear systems of the form `Ax=b`.
The simple kernel solves a single system, and is called in a loop over the batch.
The kernel sets up the system with nonsense data, although this is insignificant.

The code tests a number of options for allocating the matrix and vectors:

  1. Each thread allocates its own matrix and vector which are allocated in parallel.
  2. The master thread allocates one matrix and vector *per* thread.
  3. One memory allocation per array which is large enough to store one matrix/vector per thread.

In all cases, NUMA-aware first touch initialisation is used.

## Building

It should be simple to build the code by typing `make`.

Some default flags are included for different compilers.
The compiler can be chosen with the `COMPILER` variable, for example

    make COMPILER=INTEL


## Running

The code must be run with 3 or 4 command line arguments, detailed below:

    ./batched_gaussian_elimination order buckets ntimes [option]

    Arguments:
      order = set matrix size equal to (order+1)^3.
      buckets = size of batch, the number of systems to solve.
      ntimes = outer iteration for number of buckets to solve.
      option = optional argument. Selects which allocation option to run (default all are run).


