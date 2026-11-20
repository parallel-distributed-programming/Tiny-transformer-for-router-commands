# Build on Miyabi (as in the notebook):
#   module purge && module load nvidia/25.9 cuda/12.9 && PATH=/work/gt81/share/env/llvm/bin:$PATH
#   make
CXX       = clang++   # override with: make CXX=g++
CXXFLAGS ?= -O3 -Wall -Wextra -std=c++17

COMMON = config.o data.o ops.o model.o optimizer.o evaluate.o
HEADERS = config.h data.h ops.h model.h optimizer.h evaluate.h

all: tinygpt check

tinygpt: main.o $(COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

check: check.o $(COMMON)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o tinygpt check

.PHONY: all clean
