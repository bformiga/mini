.PHONY: test bench native native-clean
PYTHON ?= python
CXX ?= g++

test:
	$(PYTHON) -m pytest -q

bench:
	PYTHONPATH=src $(PYTHON) -m mini_sim --size 16 --rows 64 --json results/dense.json --trace results/dense.csv

native:
	$(CXX) -O3 -fPIC -shared -std=c++14 -fvisibility=hidden -ffp-contract=off -fno-fast-math -fno-associative-math -Wall -Wextra -o src/native/libatalla_kernels.so.tmp src/native/atalla_kernels.cpp
	mv src/native/libatalla_kernels.so.tmp src/native/libatalla_kernels.so

native-clean:
	rm -f src/native/libatalla_kernels.so

