#!/usr/bin/env sh
rm -r kernel.o 
#rm -r TSP.o

nvcc -L /usr/local/cuda-9.0/lib64/ -I /usr/local/cuda-9.0/samples/common/inc   -lcudart -lcuda  -c kernel.cu

echo "Done."
