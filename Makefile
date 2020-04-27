

COMPILER=INTEL

INTEL_CXX=icpc
INTEL_FLAGS=-std=c++11 -O3 -xHost -qopt-report=5 -g
INTEL_OMP=-qopenmp

CRAY_CXX=CC
CRAY_FLAGS=-std=c++11 -O3 -ffp-contract=fast -mcpu=native
CRAY_OMP=-fopenmp

GCC_CXX=g++
GCC_FLAGS=-std=c++11 -O3 -march=native
GCC_OMP=-fopenmp

batched_gaussian_elimination: batched_gaussian_elimination.cpp Makefile
	$($(COMPILER)_CXX) $($(COMPILER)_FLAGS) batched_gaussian_elimination.cpp $($(COMPILER)_OMP) -o $@

.PHONY: clean
clean:
	rm -f batched_gaussian_elimination

